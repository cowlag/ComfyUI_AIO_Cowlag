from nodes import MAX_RESOLUTION,CheckpointLoaderSimple,LoraLoader,CLIPTextEncode,EmptyLatentImage,KSampler,VAELoader,VAEDecode,VAEEncode,LoadImage,ImageScale
import folder_paths
import comfy.utils
import comfy.sd
import comfy.samplers
import torch
# MAX_RESOLUTION=8192


class txt2imgAllInOne:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """

    def __init__(self):
        self.loaded_lora = None
        self.lora_loader = None

    @staticmethod
    def vae_list():
        vaes = ["FromModel"]+folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
        return sd
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ), # CheckpointLoaderSimple
                "lora_name": (["None"]+folder_paths.get_filename_list("loras"), ), # LoraLoader
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}), # LoraLoader
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}), # LoraLoader
                "text1": ("STRING", {"multiline": True}), # CLIPTextEncode
                "text2": ("STRING", {"multiline": True}), # CLIPTextEncode
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}), # EmptyLatentImage
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}), # EmptyLatentImage
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}), # EmptyLatentImage
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # KSampler
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}), # KSampler
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}), # KSampler
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ), # KSampler
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ), # KSampler
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}), # KSampler
                "vae_name": (s.vae_list(), ) # VAELoader
            },
        }

    RETURN_TYPES = ("IMAGE",) # ("MODEL", "CLIP", "VAE","IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "txt2img"

    #OUTPUT_NODE = False

    CATEGORY = "CowlagNodes"

    def txt2img(self, 
                ckpt_name, # CheckpointLoaderSimple
                lora_name,  # LoraLoader
                strength_model,  # LoraLoader
                strength_clip, # LoraLoader
                text1, # CLIPTextEncode
                text2, # CLIPTextEncode
                width,  # EmptyLatentImage
                height,  # EmptyLatentImage
                batch_size = 1, # EmptyLatentImage
                seed=0,  # KSampler
                steps=20,  # KSampler
                cfg=8.0,  # KSampler
                sampler_name="euler",  # KSampler
                scheduler="normal",  # KSampler
                denoise=1.0, # KSampler
                vae_name="" # VAELoader
        ):

        # CheckpointLoader
        (model,clip,vae) = CheckpointLoaderSimple().load_checkpoint(ckpt_name, output_vae=True, output_clip=True)
        # return (ckpt,clip,vae)

        model_new = model
        clip_new = clip

        # LoraLoader
        if lora_name != "None":
            if self.lora_loader is None:
                self.lora_loader = LoraLoader()
            model_new, clip_new = self.lora_loader.load_lora(model, clip, lora_name, strength_model, strength_clip)
        # return (model_lora, clip_lora)

        # CLIPTextEncode
        # ([[cond, {"pooled_output": pooled}]], )
        (positive,) = CLIPTextEncode().encode(clip_new, text1)
        # return [[cond, {"pooled_output": pooled}]]
        (negative,) = CLIPTextEncode().encode(clip_new, text2)
        # return [[cond, {"pooled_output": pooled}]]



        # EmptyLatentImage
        (latent_image,) = EmptyLatentImage().generate(width, height, batch_size)
        # return {"samples":latent}


        # KSampler
        (latentoutput,) = KSampler().sample(model_new, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
        # return latentoutput


        # VAELoader
        vae_new = vae
        if vae_name != "FromModel":
            vae_new = VAELoader().load_vae(vae_name)[0]
        # return vae

        # VAEDecode
        (image,) = VAEDecode().decode(vae_new,latentoutput)
        # return image

        return (image,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


class img2imgAllInOne:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    def __init__(self):
        self.loaded_lora = None
        self.lora_loader = None

    @staticmethod
    def vae_list():
        vaes = ["FromModel"] + folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
        return sd

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            input_dir = folder_paths.get_input_directory()
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),  # CheckpointLoaderSimple
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),  # LoraLoader
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),  # LoraLoader
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),  # LoraLoader
                "text1": ("STRING", {"multiline": True}),  # CLIPTextEncode
                "text2": ("STRING", {"multiline": True}),  # CLIPTextEncode
                "upscale_method": (s.upscale_methods,), # ImageScale
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}), # ImageScale
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}), # ImageScale
                "crop": (s.crop_methods,), # ImageScale
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),  # KSampler
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),  # KSampler
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),  # KSampler
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),  # KSampler
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),  # KSampler
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),  # KSampler
                "vae_name": (s.vae_list(),),  # VAELoader
                "imageupload": (sorted(files), {"image_upload": True}) # LoadImage
            },
        }

    RETURN_TYPES = ("IMAGE",)  # ("MODEL", "CLIP", "VAE","IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "img2img"

    # OUTPUT_NODE = False

    CATEGORY = "CowlagNodes"

    def img2img(self,
                ckpt_name,  # CheckpointLoaderSimple
                lora_name,  # LoraLoader
                strength_model,  # LoraLoader
                strength_clip,  # LoraLoader
                text1,  # CLIPTextEncode
                text2,  # CLIPTextEncode
                upscale_method, # ImageScale
                width, # ImageScale
                height, # ImageScale
                crop, # ImageScale
                seed=0,  # KSampler
                steps=20,  # KSampler
                cfg=8.0,  # KSampler
                sampler_name="euler",  # KSampler
                scheduler="normal",  # KSampler
                denoise=1.0,  # KSampler
                vae_name="",  # VAELoader
                imageupload # LoadImage
                ):

        # CheckpointLoader
        (model, clip, vae) = CheckpointLoaderSimple().load_checkpoint(ckpt_name, output_vae=True, output_clip=True)
        # return (ckpt,clip,vae)

        model_new = model
        clip_new = clip

        # LoraLoader
        if lora_name != "None":
            if self.lora_loader is None:
                self.lora_loader = LoraLoader()
            model_new, clip_new = self.lora_loader.load_lora(model, clip, lora_name, strength_model, strength_clip)
        # return (model_lora, clip_lora)

        # CLIPTextEncode
        # ([[cond, {"pooled_output": pooled}]], )
        (positive,) = CLIPTextEncode().encode(clip_new, text1)
        # return [[cond, {"pooled_output": pooled}]]
        (negative,) = CLIPTextEncode().encode(clip_new, text2)
        # return [[cond, {"pooled_output": pooled}]]

        # VAELoader
        vae_new = vae
        if vae_name != "FromModel":
            vae_new = VAELoader().load_vae(vae_name)[0]
        # return vae

        # LoadImage
        (pixels,MASK,) = LoadImage().load_image(imageupload)

        # ImageScale
        (imageupscaled,) =ImageScale().upscale(pixels, upscale_method, width, height, crop)

        # VAEEncode
        (latentoutputencoded,) = VAEEncode().encode(vae_new, imageupscaled)

        # KSampler
        (latentoutput,) = KSampler().sample(model_new, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                            latentoutputencoded, denoise)
        # return latentoutput

        # VAEDecode
        (image,) = VAEDecode().decode(vae_new, latentoutput)
        # return image

        return (image,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    # @classmethod
    # def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""



# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "txt2imgAllInOne": txt2imgAllInOne,
    "img2imgAllInOne": img2imgAllInOne
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "txt2imgAllInOne": "txt2img All In One",
    "img2imgAllInOne": "img2img All In One"
}
