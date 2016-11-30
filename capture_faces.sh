max=1
pic=web_cam_shot
pic_type=.jpg
path=inp_$1/
person_name=$2_
mode=$1
echo $mode
if [ $mode == "train" ]; then max=10; fi

for i in `seq 1 $max`
do
    fswebcam -r 640x480 --jpeg 85 -D 1 $path$person_name$pic$i$pic_type
    read next
done
files=$path$person_name*
for f in $files
do
    python2  FaceDetect/face_detect.py $f  FaceDetect/haarcascade_frontalface_default.xml
    convert $f -resize 256x256 $f 
done

python2 get_features.py $1

if [ $1 == 'test' ]
    then python2 get_features.py $1; python src_temp.py
fi
