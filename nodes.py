import comfy
import comfy.utils
import comfy.text_encoders
import folder_paths
import logging
from comfy.sd import CLIP

from .gemma import LuminaGemmaTokenizer, LuminaGemmaClip

class GemmaCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("clip"), ),
                             }}
    TITLE = "Gemma CLIP Loader"
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name):
        class EmptyClass:
            pass

        model_options = {}
        tokenizer_data = {}

        clip_path = folder_paths.get_full_path_or_raise("clip", clip_name)
        clip_data = []
        clip_data.append(comfy.utils.load_torch_file(clip_path, safe_load=True))

        parameters = 0
        for c in clip_data:
            parameters += comfy.utils.calculate_parameters(c)
            tokenizer_data, model_options = comfy.text_encoders.long_clipl.model_options_long_clip(c, tokenizer_data, model_options)

        clip_target = EmptyClass()
        clip_target.params = {}
        clip_target.clip = LuminaGemmaClip
        clip_target.tokenizer = LuminaGemmaTokenizer
        clip = CLIP(clip_target, embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    parameters=parameters,
                    tokenizer_data=tokenizer_data,
                    model_options=model_options)
        for c in clip_data:
            m, u = clip.load_sd({k.partition('model.')[2] if 'model.' in k else k: c[k] for k in c.keys()})
            if len(m) > 0:
                logging.warning("clip missing: {}".format(m))

            if len(u) > 0:
                logging.debug("clip unexpected: {}".format(u))
        return (clip,)

NODE_CLASS_MAPPINGS = {
    "GemmaClipLoader": GemmaCLIPLoader
}
