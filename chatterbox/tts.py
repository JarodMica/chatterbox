from dataclasses import dataclass
from pathlib import Path
import logging

import librosa
import numpy as np
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.metrics.pairwise import cosine_similarity

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.t3.modules.t3_config import T3Config


REPO_ID = "ResembleAI/chatterbox"


def smart_load_t3_model(t3_state_dict, device="cpu"):
    """
    Smart loading function that automatically detects and adjusts T3Config 
    based on the checkpoint dimensions, particularly text_tokens_dict_size.
    """
    logger = logging.getLogger(__name__)
    
    # Try loading with default config first
    config = T3Config()
    t3 = T3(config)
    
    try:
        t3.load_state_dict(t3_state_dict)
        logger.info(f"Successfully loaded T3 model with default config (text_tokens_dict_size={config.text_tokens_dict_size})")
        return t3.to(device).eval()
    except RuntimeError as e:
        error_msg = str(e)
        logger.warning(f"Initial loading failed with default config: {error_msg}")
        
        # Check if it's a size mismatch error for text embeddings
        if "text_emb.weight" in error_msg and "size mismatch" in error_msg:
            # Parse the error to extract the correct size
            import re
            # Look for pattern like "torch.Size([704, 1024])" in the error message
            checkpoint_size_match = re.search(r'copying a param with shape torch\.Size\(\[(\d+), \d+\]\)', error_msg)
            
            if checkpoint_size_match:
                correct_text_tokens_dict_size = int(checkpoint_size_match.group(1))
                logger.info(f"Detected correct text_tokens_dict_size from checkpoint: {correct_text_tokens_dict_size}")
                
                # Create new config with correct size
                corrected_config = T3Config()
                corrected_config.text_tokens_dict_size = correct_text_tokens_dict_size
                
                # Create new model with corrected config
                t3_corrected = T3(corrected_config)
                
                try:
                    t3_corrected.load_state_dict(t3_state_dict)
                    logger.info(f"Successfully loaded T3 model with corrected config (text_tokens_dict_size={correct_text_tokens_dict_size})")
                    return t3_corrected.to(device).eval()
                except RuntimeError as retry_error:
                    logger.error(f"Failed to load even with corrected config: {retry_error}")
                    raise retry_error
            else:
                logger.error(f"Could not parse checkpoint size from error message: {error_msg}")
                raise e
        else:
            # Different type of error, re-raise
            logger.error(f"Non-size-mismatch error during loading: {error_msg}")
            raise e


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()
        
    def _redact_bracketed_audio(self, wav, text):
        """
        Post-processing approach similar to Tortoise TTS redaction.
        Removes audio segments corresponding to bracketed text using wav2vec2 alignment.
        """
        if '[' not in text:
            return wav
            
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
            
            # Initialize wav2vec2 models (same as Tortoise)
            model = Wav2Vec2ForCTC.from_pretrained("jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli").to(self.device)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('jbetker/tacotron-symbols')
            
            # Parse bracketed sections
            splitted = text.split('[')
            fully_split = [splitted[0]]
            for spl in splitted[1:]:
                if ']' not in spl:
                    print(f"Warning: Found '[' without matching ']' in text: {text}")
                    return wav
                fully_split.extend(spl.split(']'))
            
            # Identify non-redacted intervals (text we want to keep)
            non_redacted_intervals = []
            last_point = 0
            for i in range(len(fully_split)):
                if i % 2 == 0 and fully_split[i] != "":  # Keep even indices (non-bracketed)
                    end_interval = max(0, last_point + len(fully_split[i]) - 1)
                    non_redacted_intervals.append((last_point, end_interval))
                last_point += len(fully_split[i])
            
            if not non_redacted_intervals:
                print("Warning: No non-bracketed text found to keep")
                return wav
            
            # Create alignment using wav2vec2
            bare_text = ''.join(fully_split)  # Text without brackets
            alignments = self._align_audio_to_text(wav, bare_text, model, tokenizer)
            print(alignments)
            # Extract and concatenate non-redacted audio segments
            output_audio = []
            for start_char, end_char in non_redacted_intervals:
                if start_char < len(alignments) and end_char < len(alignments):
                    start_sample = alignments[start_char]
                    end_sample = alignments[end_char] if end_char < len(alignments) - 1 else wav.shape[-1]
                    output_audio.append(wav[:, start_sample:end_sample])
            
            if output_audio:
                redacted_wav = torch.cat(output_audio, dim=-1)
                print(f"Redaction successful: {wav.shape[-1]} -> {redacted_wav.shape[-1]} samples")
                return redacted_wav
            else:
                print("Warning: No audio segments to concatenate")
                return wav
                
        except Exception as e:
            print(f"Redaction failed: {e}, returning original audio")
            return wav
    
    def _max_alignment(self, s1, s2, skip_character='~', record=None):
        """
        Tortoise's dynamic programming alignment algorithm.
        Aligns s1 to s2 as best it can, using '~' for unmatched characters.
        """
        if record is None:
            record = {}
        assert skip_character not in s1, f"Found the skip character {skip_character} in the provided string, {s1}"
        if len(s1) == 0:
            return ''
        if len(s2) == 0:
            return skip_character * len(s1)
        if s1 == s2:
            return s1
        if s1[0] == s2[0]:
            return s1[0] + self._max_alignment(s1[1:], s2[1:], skip_character, record)

        take_s1_key = (len(s1), len(s2) - 1)
        if take_s1_key in record:
            take_s1, take_s1_score = record[take_s1_key]
        else:
            take_s1 = self._max_alignment(s1, s2[1:], skip_character, record)
            take_s1_score = len(take_s1.replace(skip_character, ''))
            record[take_s1_key] = (take_s1, take_s1_score)

        take_s2_key = (len(s1) - 1, len(s2))
        if take_s2_key in record:
            take_s2, take_s2_score = record[take_s2_key]
        else:
            take_s2 = self._max_alignment(s1[1:], s2, skip_character, record)
            take_s2_score = len(take_s2.replace(skip_character, ''))
            record[take_s2_key] = (take_s2, take_s2_score)

        return take_s1 if take_s1_score > take_s2_score else skip_character + take_s2

    def _align_audio_to_text(self, audio, expected_text, model, tokenizer):
        """Full Tortoise-style wav2vec2 alignment implementation"""
        orig_len = audio.shape[-1]
        
        with torch.no_grad():
            # Resample to 16kHz for wav2vec2
            import torchaudio
            audio_16k = torchaudio.functional.resample(audio, self.sr, 16000)
            
            # Normalize audio
            clip_norm = (audio_16k - audio_16k.mean()) / torch.sqrt(audio_16k.var() + 1e-7)
            logits = model(clip_norm.to(self.device)).logits

        logits = logits[0]
        pred_string = tokenizer.decode(logits.argmax(-1).tolist())
        
        # Use Tortoise's max_alignment algorithm
        fixed_expectation = self._max_alignment(expected_text.lower(), pred_string)
        w2v_compression = orig_len // logits.shape[0]
        expected_tokens = tokenizer.encode(fixed_expectation)
        expected_chars = list(fixed_expectation)
        
        if len(expected_tokens) == 1:
            return [0]  # Simple case
        expected_tokens.pop(0)  # Remove first token
        expected_chars.pop(0)

        alignments = [0]
        def pop_till_you_win():
            if len(expected_tokens) == 0:
                return None
            popped = expected_tokens.pop(0)
            popped_char = expected_chars.pop(0)
            while popped_char == '~':
                alignments.append(-1)
                if len(expected_tokens) == 0:
                    return None
                popped = expected_tokens.pop(0)
                popped_char = expected_chars.pop(0)
            return popped

        next_expected_token = pop_till_you_win()
        for i, logit in enumerate(logits):
            top = logit.argmax()
            if next_expected_token == top:
                alignments.append(i * w2v_compression)
                if len(expected_tokens) > 0:
                    next_expected_token = pop_till_you_win()
                else:
                    break

        pop_till_you_win()
        if not (len(expected_tokens) == 0 and len(alignments) == len(expected_text)):
            print(f"Alignment warning: expected {len(expected_text)} alignments, got {len(alignments)}")
            # Fall back to linear alignment
            return [int(i * orig_len / len(expected_text)) for i in range(len(expected_text))]

        # Fix up alignments - interpolate -1 values
        alignments.append(orig_len)  # Temporary for algorithm
        for i in range(len(alignments)):
            if alignments[i] == -1:
                for j in range(i+1, len(alignments)):
                    if alignments[j] != -1:
                        next_found_token = j
                        break
                for j in range(i, next_found_token):
                    gap = alignments[next_found_token] - alignments[i-1]
                    alignments[j] = (j-i+1) * gap // (next_found_token-i+1) + alignments[i-1]

        return alignments[:-1]  # Remove temporary last element

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        # Load T3 model with smart loading
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        # if "model" in t3_state.keys():
        #     t3_state = t3_state["model"][0]
        if any(k.startswith("t3.") for k in t3_state):
            t3_state = {k[len("t3."):]: v for k, v in t3_state.items()}
        
        t3 = smart_load_t3_model(t3_state, device)

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)
    
    @classmethod
    def from_specified(
        cls,
        voice_encoder_path,
        t3_path,
        s3gen_path,
        tokenizer_path,
        conds_path,
        device
    ):
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None
            
        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(voice_encoder_path)
        )
        ve.to(device).eval()

        # Load T3 model with smart loading
        t3_state = load_file(t3_path)
        # if "model" in t3_state.keys():
        #     t3_state = t3_state["model"][0]
        if any(k.startswith("t3.") for k in t3_state):
            t3_state = {k[len("t3."):]: v for k, v in t3_state.items()}
        
        t3 = smart_load_t3_model(t3_state, device)

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(s3gen_path), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(tokenizer_path)
        )

        conds = None
        if (builtin_voice := conds_path).exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, 
                           translate_to=None, translation_strength=0.7, 
                           translation_model_path="Voice_embeddings_study/voice_translator_simplified.pt",
                           source_language=None, force_translation=False):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True)
        
        if translate_to is not None:
            ve_embed = self._translate_voice_embedding(
                ve_embed.squeeze(0).cpu().numpy(), 
                translate_to, 
                translation_strength, 
                translation_model_path,
                source_language=source_language,
                force_translation=force_translation
            )
            ve_embed = torch.from_numpy(ve_embed).unsqueeze(0)
        
        ve_embed = ve_embed.to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def _translate_voice_embedding(self, embedding, target_language, strength, model_path, 
                                   source_language=None, force_translation=False, 
                                   similarity_threshold=0.05):
        data = torch.load(model_path, map_location='cpu')
        language_centers = {lang: center.numpy() for lang, center in data["language_centers"].items()}
        
        if source_language is None:
            similarities = {}
            for lang, center in language_centers.items():
                sim = cosine_similarity([embedding], [center])[0][0]
                similarities[lang] = sim
            
            source_language = max(similarities, key=similarities.get)
            
            sorted_sims = sorted(similarities.values(), reverse=True)
            if len(sorted_sims) > 1:
                similarity_diff = sorted_sims[0] - sorted_sims[1]
                if similarity_diff < similarity_threshold:
                    print(f"Warning: Language detection uncertain. Similarity difference: {similarity_diff:.4f}")
                    if not force_translation:
                        print(f"Detected language: {source_language}, but confidence is low. Consider setting source_language explicitly.")
        
        if source_language == target_language and not force_translation:
            return embedding
                
        if target_language not in language_centers:
            available_languages = list(language_centers.keys())
            raise ValueError(f"Target language '{target_language}' not available. Available languages: {available_languages}")
        
        target_center = language_centers[target_language]
        source_center = language_centers[source_language]
        translation_vector = target_center - source_center
        translated = embedding + strength * translation_vector
        translated = translated / np.linalg.norm(translated)
        
        return translated

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        redact=False,
        translate_to=None,
        translation_strength=0.7,
        source_language=None,
        force_translation=True,
        translation_model_path="voice_translator_simplified.pt"
    ):
        if audio_prompt_path:
            self.prepare_conditionals(
                audio_prompt_path, 
                exaggeration=exaggeration, 
                translate_to=translate_to, 
                translation_strength=translation_strength,
                source_language=source_language,
                force_translation=force_translation,
                translation_model_path=translation_model_path
            )
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        if '[' in text:
            text_1 = text.replace('[', '').replace(']', '')
        else:
            text_1 = text
        text_tokens = self.tokenizer.text_to_tokens(text_1).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            result = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            
            # Handle both old format (tensor) and new format (dict)
            if isinstance(result, dict):
                speech_tokens = result['final_tokens']
                first_gen_tokens = result.get('first_generation')
                
                # Save first generation audio if it exists
                if first_gen_tokens is not None:
                    try:
                        # Extract only the conditional batch for first generation
                        first_tokens = first_gen_tokens[0] if first_gen_tokens.dim() > 1 else first_gen_tokens
                        first_tokens = drop_invalid_tokens(first_tokens)
                        first_tokens = first_tokens.to(self.device)
                        
                        # Generate audio for first generation
                        first_wav, _ = self.s3gen.inference(
                            speech_tokens=first_tokens,
                            ref_dict=self.conds.gen,
                        )
                        first_wav = first_wav.squeeze(0).detach().cpu().numpy()
                        
                        # Save to seq_debug.wav
                        import torchaudio
                        torchaudio.save("seq_debug.wav", torch.from_numpy(first_wav).unsqueeze(0), self.sr)
                        print("DEBUG: Saved first generation audio to seq_debug.wav")
                        
                    except Exception as e:
                        print(f"DEBUG: Could not save first generation audio: {e}")
            else:
                # Old format - just a tensor
                speech_tokens = result
            
            # Note: speech_tokens contain full generation including BL/EL content
            # Audio-level redaction will be applied after synthesis
            
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0] if speech_tokens.dim() > 1 else speech_tokens

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            wav_tensor = torch.from_numpy(wav).unsqueeze(0)
            
            # Apply redaction if bracketed text is present
            if redact:
                wav_tensor = self._redact_bracketed_audio(wav_tensor, text)
            
            # Apply watermark
            watermarked_wav = self.watermarker.apply_watermark(wav_tensor.squeeze(0).numpy(), sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)