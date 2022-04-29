source /home/guest/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
path="/home/guest/Android-Smoke-app/Android_Module/smokePredict"
io_path="/home/guest/Android-Smoke-app/img_out"
cd $path
command="python predict.py predict ${1} ${io_path}/img_out ${io_path}/smoke_out "
result=`$command`
echo $result
conda deactivate