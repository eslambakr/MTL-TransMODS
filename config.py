class Config:
    # Choose input format for kitti-tracking dataset
    # Choose one of the following:
    # [baseline, stacking, shared_backbone, shared_rgb_of, spatial_stacking, depth_pos_enc, cat_4frames_res34,
    # depth_pos_enc_arch2, depth_pos_enc_arch4, shared_rgb_of_N]
    # Notes:
    # 1) "depth_pos_enc_arch1" is (depth_pos_enc + encode_channels)
    #   1.1) has different variant by changing TPE_type, Arch#1 --> ["HW_NC", "NHW_C", "NC_HW"]
    #        For "NHW_C" disable channel encoding as I am using 3d pos encoding
    # 2) For "depth_pos_enc_arch2", a specific training on coco was made to generate full model weights
    #    to include pre-trained weights for 1*1 Conv.
    #    3.1) "NB_NTE_1T":
    #         Shared TE is used then fuse them using another TE then pass them to TD to generate Object Queries.
    #         "encode_channels" must set to False as this variant be default is using TPE
    #         Use the same pre-trained weights on coco like "depth_pos_enc_arch2"
    # 3) "depth_pos_enc_arch3" is like "depth_pos_enc_arch2" with sharing = False.
    #
    # 4) "depth_pos_enc_arch4" consists of two separate DETRs for each input stream and one DETR to fuse both of them.
    #    - Pre-training on COCO dataset was used to initialize transformers weights but using only one stream
    #      and use a trick to duplicate the weights and use it for the both streams (backbone+Transformer)
    #    - Tracking/Moving object detection data should be used instead of COCO for the pre-training phase [In progress]
    #    - It has "encode_channels" variants and also "concat" variants, set them True to activate them.
    #    - Please Note that encoding channels couldn't be used while concat is set to false.
    # 5) "shared_rgb_of_N" repeat the one branch of "shared_rgb_of" experiment, N times
    #    - N should be >= 2
    #    - Variants from this Exp.:
    #       1- "N_B_1_T"
    #       2- "N_B_N_T"
    #       3- "N_B_N_T_1_T"
    #       4- "N_B_proj_1_T"   ,N backbones then fuse them using 1*1 Conv. and feed them to Transformer.
    #                            Note that "concat" must be activated, support "encode_channels" feature.
    #       5- "N_B_N_proj_1_T"   ,N backbones then fuse them to reduce C using 1*1 Conv
    #                              and feed them after concatenation to a Transformer.
    #                              Note that "concat" must be activated, support "encode_channels" feature.
    #       6- "N_B_N_T_proj_1_T"   ,N backbones and N Transformers (SPE) then fuse them to reduce C using 1*1 Conv
    #                              and feed them after concatenation to a Transformer (TPE).
    #                              Note that "concat" must be activated.
    #                              Use "N_B_proj_1_T" as a pre-trained weights while training on coco.
    #                              Use "log_coco_4B_4T_4proj_1T" as a pre-trained weights while training on kittiold.
    #       7- "N_B_proj_1_T_de"    , it is exactly like "N_B_proj_1_T" but I have decoupled the current frame
    #                                 from prev. ones, by treating the current frame as the Query
    #                                 and the current+prev. as Keys.
    #    - N shared backbones and N shared Transformers.

    # 6) "shared_backbone_rgb_of_N", In the Exp. will repeat backbone only N times and fuse their output(Add/Concat)
    #    then feed them to one transformer.
    exp_type = "shared_rgb_of_N"

    # Parameters dedicated for Exp. "shared_rgb_of_N" & "depth_pos_enc_arch2" only!:
    variant = "N_B_N_proj_1_T"
    # parameter dedicated for "N_B_N_T" variant.  [concat, FC, NQ_C, NQ_C_Decoder]
    # parameter dedicated for "N_B_proj_1_T" variant.  [decouple_curr_prev, decouple_curr_repeated_prev]
    # parameter dedicated for "N_B_N_proj_1_T" variant.  [decouple_curr_prev, decouple_curr_repeated_prev]
    sub_variant = ""
    num_fusion_layers = 3
    aux_q = False  # Set it to True, To Activate the AUX loss for the N queries that are produced from N transformers.

    sharing = False  # dedicated to variant "N_B_N_T_proj_1_T" which means share the transformers or not
    # and dedicated to variant "N_B_N_T +NQ_C/NQ_C_Decoder" which means share the AUX Q output heads.

    # dedicated to arch#2&3, if set to True then Arch#2 is activated if set to False then arch#3 is activated.
    num_of_repeated_blocks = 2  # Set it when activating exp. = "shared_rgb_of_N"

    # Parameters for Exp. Arch#1, Arch#2 --> ["HW_NC", "NHW_C", "NC_HW", "N_CHW"]
    TPE_type = ""

    load_full_model = False
    pre_training_coco = True

    # If and only if "depth_pos_enc" or "depth_pos_enc_arch2" was chosen: activate or deactivate this parameter:
    encode_channels = True

    # If concat is enabled the output feature map of backbones will be concatenated with each others
    # before feed them to transformer
    # If Disabled the backbones features will be added with each others
    concate = True

    augment = False
    resize_with_aspect_ratio = False
    input_size = 512
    num_classes = None
    enable_instance_seg = False

    # Backbone
    Backbone_types = ["DETR", "ViT"]
    backbone_type = Backbone_types[0]
    # In case of ViT:
    ViT_learnable_pos = False  # If True the positional encoding will be learnable if not it will be Sinusoidal
    patch_size = 16
    backbone_channels = 64

    # MTL parameters
    MTL = True
    det_task_status = True
    seg_task_status = True
    shared_dec_concat_q = False
    shared_dec_shared_q = True

    # Segmentation Parameters:
    if (not MTL) and seg_task_status:
        seg_only = True
    else:
        seg_only = False
    convert_coco_to1class = False

    # Testing Parameters:
    saving_det_out = False
    _saving_dir = "log_MTL_Det_Seg_512_kittiold_scal=100_shared_dec/"
    saving_det_out_dir = _saving_dir + "out_imgs/"
    saving_attention_map = False
    saving_attention_map_dir = _saving_dir + "out_attention/"

    # Inference Parameters:
    save_att_maps_per_object = False
    save_att_maps_per_img = True
    saving_out_RGB = True
    conf_th = 0.9
