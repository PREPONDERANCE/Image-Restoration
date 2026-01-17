##############################################################
############   Test on LOLv2 Enhancement Datset   ############
##############################################################

# LOLv2 Syn For AST
python -m Enhancement.test_from_dataset_LOLv2_Syn \
    --opt './Enhancement/Options/AST_LOL_v2_synthetic.yml' \
    --result_dir './results/AST' \
    --weights './experiments/Enhancement_AST/models/net_g_latest.jth' \
    --gpus 0

# LOLv2 Syn for Hint
python -m Enhancement.test_from_dataset_LOLv2_Syn \
    --opt './Enhancement/Options/HINT_LOL_v2_synthetic.yml' \
    --weights './pretrained/LOLv2_Syn_Hint.jth' \
    --gpus 0

# LOLv2 Syn for ASTv2
python -m Enhancement.test_from_dataset_LOLv2_Syn \
    --opt './Enhancement/Options/ASTV2_LOL_v2_synthetic.yml' \
    --weights '' \
    --gpus 0

# LOLv2 Real for Hint
python -m Enhancement.test_from_dataset_LOLv2_Real \
    --opt './Enhancement/Options/HINT_LOL_v2_real.yml' \
    --weights './pretrained/LOLv2_Real_Hint.jth' \
    --gpus 0

##############################################################
##############################################################
##############################################################


##############################################################
#############    Test on SOTS Dehazing Datset    #############
##############################################################
 
# FPro
python -m Dehaze.test_SOTS \
    --opt './Dehaze/Options/RealDehazing_FPro.yml' \
    --result_dir './results/FPro' \
    --weights './pretrained/SOTS_FPro.pkl' \
    --gpus 0

python -m Dehaze.evaluate_SOTS \
    --result_dir './results/FPro/outdoor' \
    --gt_dir '/home/ubuntu/gwt/data/promptIR/outdoor/gt'
    # Replace with your own promptIR outdoor dataset path

##############################################################
##############################################################
##############################################################