# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import time
import sys
import json
import cv2
import numpy as np
from PIL import Image

from network import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的smoke3.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    # ----------------------------------------------------------------------------------------------------------#
    mode = sys.argv[1]
    #mode = "video"
    # -------------------------------------------------------------------------#
    #   picture_save_path表示predict模式下检测后的图片存放路径
    #   smoke_save_path表示predict模式下检测后裁剪的烟雾存放路径
    # -------------------------------------------------------------------------#
    picture_save_path = sys.argv[3]
    smoke_save_path = sys.argv[4]
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_save_path = "video_out/smoke2.mp4"
    video_smoke_save_path = "smoke_out/"
    video_fps = 19.5
    # -------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    # -------------------------------------------------------------------------#
    test_interval = 100
    # -------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_smoke_save_path指定了裁剪烟雾的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    dir_smoke_save_path = "smoke_out/"

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        import os

        img = sys.argv[2]
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            exit(0)
        else:
            r_image, smokes, levels, nums = yolo.detect_image(image)
            r_image.show()
            smokeJson = []
            img_allname = img.split('/')[-1]
            img_name = img_allname.split('.')[0]
            picture_save_path = picture_save_path + '/'
            if not os.path.exists(picture_save_path):
                 os.makedirs(picture_save_path)
            r_image.save(os.path.join(picture_save_path,img_allname))
            the_path = picture_save_path + img_allname
            
            smokeDict = dict(dir=the_path)
            # js = json.dumps(smokeDict)
            smokeJson.append(smokeDict)
            
            smoke_save_path = smoke_save_path + '/' + img_name + '/'
            if not os.path.exists(smoke_save_path):
                os.makedirs(smoke_save_path)
            for i in range(nums):
                the_path = img_name + "_" + str(i + 1) + '.' + img_allname.split('.')[1]
                smokes[i].save(os.path.join(smoke_save_path, the_path))
                the_path = smoke_save_path + the_path
                smokeDict = dict(dir=the_path)
                # js = json.dumps(smokeDict)
                smokeJson.append(smokeDict)
            result = dict(data = smokeJson , res_dir = smoke_save_path)
            print(json.dumps(result))
            

    elif mode == "video":
        import os

        video_path = sys.argv[2]
        #video_path = "video/smoke1.mp4"
        capture = cv2.VideoCapture(video_path)
        smokeJson = []
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
            smokeDict = dict(dir=video_save_path)
            js = json.dumps(smokeDict, indent=4)
            smokeJson.append(js)

        fps = 0.0
        j = 0
        while (True):
            j+=1
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if frame is None:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame, smokes, levels, nums = yolo.detect_image(frame)
            frame = np.array(frame)
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            #print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
            if nums == 0:
                continue
            for i in range(nums):
                the_path = "videoSmoke" + str(j) + "_" + str(i + 1) + ".jpg"
                smokes[i].save(os.path.join(video_smoke_save_path, the_path))
                the_path = video_smoke_save_path + the_path
                smokeDict = dict(dir=the_path)
                js = json.dumps(smokeDict, indent=4)
                smokeJson.append(js)
        capture.release()
        out.release()
        cv2.destroyAllWindows()
        print(smokeJson)

    elif mode == "fps":
        img = Image.open('img/smoke3.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        #print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image, smokes, levels, nums = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

                if nums == 0:
                    continue

                if not os.path.exists(dir_smoke_save_path):
                    os.makedirs(dir_smoke_save_path)

                for i in range(nums):
                    the_path = str(i+1) + "_" + img_name
                    smokes[i].save(os.path.join(dir_smoke_save_path, the_path))

    elif mode == "predictContinue":
        import os
        while True:
            img = input('Input image filename:')
            if img == '0':
                exit(0)
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image, smokes, levels, nums = yolo.detect_image(image)
                r_image.show()
                smokeJson = []
                if not os.path.exists(picture_save_path):
                    os.makedirs(picture_save_path)
                r_image.save(os.path.join(picture_save_path, img))
                the_path = picture_save_path + img
                smokeDict = dict(dir=the_path)
                js = json.dumps(smokeDict, indent=4)
                smokeJson.append(js)

                if nums == 0:
                    continue

                if not os.path.exists(smoke_save_path):
                    os.makedirs(smoke_save_path)

                for i in range(nums):
                    the_path = str(i + 1) + "_" + img
                    smokes[i].save(os.path.join(smoke_save_path, the_path))
                    the_path = smoke_save_path + the_path
                    smokeDict = dict(dir=the_path)
                    js = json.dumps(smokeDict, indent=4)
                    smokeJson.append(js)
                print(smokeJson)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
