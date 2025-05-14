#MAX_DELAY = 255
MAX_DELAY = 255
###############
#LOSS = "EXP"

################
QUANTIZE = False
scale = 1
p_scale = 1
w_scale = 1
s_scale = 1

class loss_config:
    LOSS = "THR"
    def change_loss(loss):
        loss_config.LOSS = loss
