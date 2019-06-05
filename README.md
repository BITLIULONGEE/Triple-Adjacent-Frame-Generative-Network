# Triple-Adjacent-Frame-Generative-Network
Simple Tensorpack implementation of "Triple-Adjacent-Frame Generative Network for Blind Video Motion Deblurring " (TAFDeblurGAN)
# for training
python train.py

#If you want to continue train, before you train, you should change the START_EPOCH into the breakpoint in the train.py
run python train.py --continue --load $$The path of the checkpoints + the name of the last of DATA-00000-OF-00001 file$$
# for testing

#the test example is formated by 2×2 images
# |---the previous frame---|---the current frame---|
# |----the last frame------|---the ground truth----|
 
run python test.py  --input_path $$*******$$   --output_path $$******$$
 
 #--input_path: the path of your testing example
 #--output_path: the path of saved deblurred images
