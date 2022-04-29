source /home/guest/anaconda3/etc/profile.d/conda.sh
conda activate paddle
path="/home/guest/Android-Smoke-app/Android_Module/PaddleSeg"
cd $path
command="python predict.py --config configs/ocrnet/ocrnet_hrnetw48_voc12aug_512x512_40k.yml --model_path output/best_model/model.pdparams --image_path ${1} --save_dir ${2}"
result=`$command`
echo $result
conda deactivate