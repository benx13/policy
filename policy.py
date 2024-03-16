import cv2
import coremltools
import time
from utils import *  
from tqdm import tqdm
import yaml
import datetime
from post import Post
import os

class policy_plot:
    def __init__(self, config_dir, source ,cap_skip=0, live=False) -> None:
        self.live = live
        self.posts = self.init_posts(config_dir)
        self.configs = self.init_plot_configs(config_dir)
        self.model = coremltools.models.MLModel('models/blue_feb_3X++.mlpackage')
        self.init_cv2_window()
        self.cap = self.init_cap(source, cap_skip)
        if live==False and type(source) == str and source.split('.')[-1] in ['mov', 'MOV', 'mp4', 'avi']:
            self.get_current_time = lambda: frame_to_hms(self.cap.get(cv2.CAP_PROP_POS_FRAMES), self.cap_fps)
            self.cap_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))-10
            self.cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.get_current_time = lambda: datetime.datetime.now().strftime("%H:%M:%S")

    def init_cv2_window(self):
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                cv2.setWindowTitle('Window', f'Coordinates: ({x}, {y})')
        cv2.namedWindow("Window", cv2.WINDOW_NORMAL) 
        cv2.setMouseCallback('Window', mouse_callback)

    def init_posts(self, config_dir):
        posts = []
        for config in sorted(os.listdir(config_dir)):
            posts.append(Post(f'{config_dir}/{config}'))
        return posts
    
    def init_plot_configs(self, config_dir):
        configs = []
        for config in sorted(os.listdir(config_dir)):      
            with open(f'{config_dir}/{config}') as f:
                configs.append(yaml.load(f, Loader=yaml.FullLoader))
        return configs
    
    def init_cap(self, source, cap_skip):
        cap = cv2.VideoCapture(source)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        if cap_skip != 0:
            skip_frames = int(frame_rate * cap_skip)
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
        return cap
    def run_model(self, input):
        return self.model.predict({'image': input,
                                'iouThreshold': 0.8, 
                                'confidenceThreshold': 0.3})
    
    def preprocess_frame(self, img):
        blues = get_blue(img)
        resized_img = cv2.resize(blues, (640, 640))
        return preprocess_img(resized_img)
    
    def close_cap(self, current_time):
        for post in self.posts:
            post.close(current_time)
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        if self.live:
            self.run_cam()
        else:
            self.run_vid()


    def run_cam(self):
        while True:
            start = time.time()
            img = self.cap.read()[1]
            current_time = self.get_current_time()

            plot_time_on_frame(img, current_time)

            for post in self.posts:
                post.reset
                post.update_transition(img)

            input = self.preprocess_frame(img)
            mstart = time.time()
            results = self.run_model(input)
            mstop = time.time()

            for confidence, (xn, yn, widthn, heightn) in zip(results['confidence'], results['coordinates']):
                x, y, x1, y1, x2, y2 = get_coordinates(img, xn, yn, widthn, heightn)
                plot_rectangles1(img, x1,y1,x2,y2,confidence[0])
                for post in self.posts:
                    post.update_objects((x, y, x1, y1, x2, y2))


            for post in self.posts:
                post.forward(current_time)
                post.logger.update_logs()
                plot_logs(img, (0, 700), post.logger.logs, 'POST1')

            for config in self.configs:
                overlay_region(img, config['TRANSITION_ZONE_01'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_02'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_03'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_04'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_05'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_06'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_07'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_08'], alpha=0.5)
            
            imgX = img.copy()

            for config in self.configs:
                overlay_region(imgX, config['GRAB_ZONE'], alpha=0.5)
                overlay_region(imgX, config['GRAB_ZONE_2'], alpha=0.5)
                overlay_region(imgX, config['FORWARD_ZONE'], alpha=0.5)
                overlay_region(imgX, config['BACKWARD_ZONE'], alpha=0.5)
                overlay_region(imgX, config['MACHINE_ZONE'], alpha=0.5)

            cv2.imshow('WindowX', img)
            cv2.imshow('Window', imgX)
            #------------------
            stop = time.time()
            inference = mstop - mstart
            total = stop - start
            #------------------
            print(f'total time:{total*1000} inference time: {inference*1000}')
            for post in self.posts:
                print(f'POST1: Count: {post.logger.stats["BAG_COUNT"]}')
                print(f'POST1: Bag time mean: {post.logger.stats["t2t_mean"]}')
                print(f'POST1: Clean bag time mean: {post.logger.stats["average_time_per_bag"]}')
            print(20*'-')
            #------------------
            if cv2.waitKey(1) == ord('q'):
                break
        self.close_cap(current_time)


    def run_vid(self):
        for _ in tqdm(range(self.cap_length)):
            start = time.time()
            img = self.cap.read()[1]
            current_time = self.get_current_time()

            plot_time_on_frame(img, current_time)

            for post in self.posts:
                post.reset
                post.update_transition(img)

            input = self.preprocess_frame(img)
            mstart = time.time()
            results = self.run_model(input)
            mstop = time.time()

            for confidence, (xn, yn, widthn, heightn) in zip(results['confidence'], results['coordinates']):
                x, y, x1, y1, x2, y2 = get_coordinates(img, xn, yn, widthn, heightn)
                plot_rectangles1(img, x1,y1,x2,y2,confidence[0])
                for post in self.posts:
                    post.update_objects((x, y, x1, y1, x2, y2))


            for post in self.posts:
                post.forward(current_time)
                post.logger.update_logs()
                plot_logs(img, (0, 700), post.logger.logs, 'POST1')

            for config in self.configs:
                overlay_region(img, config['TRANSITION_ZONE_01'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_02'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_03'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_04'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_05'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_06'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_07'], alpha=0.5)
                overlay_region(img, config['TRANSITION_ZONE_08'], alpha=0.5)
            
            imgX = img.copy()

            for config in self.configs:
                overlay_region(imgX, config['GRAB_ZONE'], alpha=0.5)
                overlay_region(imgX, config['GRAB_ZONE_2'], alpha=0.5)
                overlay_region(imgX, config['FORWARD_ZONE'], alpha=0.5)
                overlay_region(imgX, config['BACKWARD_ZONE'], alpha=0.5)
                overlay_region(imgX, config['MACHINE_ZONE'], alpha=0.5)

            cv2.imshow('WindowX', img)
            cv2.imshow('Window', imgX)
            #------------------
            stop = time.time()
            inference = mstop - mstart
            total = stop - start
            #------------------
            print(f'total time:{total*1000} inference time: {inference*1000}')
            for post in self.posts:
                print(f'POST1: Count: {post.logger.stats["BAG_COUNT"]}')
                print(f'POST1: Bag time mean: {post.logger.stats["t2t_mean"]}')
                print(f'POST1: Clean bag time mean: {post.logger.stats["average_time_per_bag"]}')
            print(20*'-')
            #------------------
            if cv2.waitKey(1) == ord('q'):
                break
        self.close_cap(current_time)

class policy_noplot:
    def __init__(self, config_dir, source ,cap_skip=0, live=False) -> None:
        self.live = live
        self.posts = self.init_posts(config_dir)
        self.model = coremltools.models.MLModel('models/blue_feb_3X++.mlpackage')
        self.cap = self.init_cap(source, cap_skip)
        if live==False and type(source) == str and source.split('.')[-1] in ['mov', 'MOV', 'mp4', 'avi']:
            self.get_current_time = lambda: frame_to_hms(self.cap.get(cv2.CAP_PROP_POS_FRAMES), self.cap_fps)
            self.cap_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))-10
            self.cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.get_current_time = lambda: datetime.datetime.now().strftime("%H:%M:%S")

    def init_posts(self, config_dir):
        posts = []
        for config in os.listdir(config_dir):
            posts.append(Post(f'{config_dir}/{config}'))
        return posts
    
    def init_cap(self, source, cap_skip):
        cap = cv2.VideoCapture(source)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        if cap_skip != 0:
            skip_frames = int(frame_rate * cap_skip)
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
        return cap
    def run_model(self, input):
        return self.model.predict({'image': input,
                                'iouThreshold': 0.8, 
                                'confidenceThreshold': 0.3})
    
    def preprocess_frame(self, img):
        blues = get_blue(img)
        resized_img = cv2.resize(blues, (640, 640))
        return preprocess_img(resized_img)
    
    def close_cap(self, current_time):
        for post in self.posts:
            post.close(current_time)
        self.cap.release()

    def run(self):
        if self.live:
            self.run_cam()
        else:
            self.run_vid()

    def run_vid(self):
        for _ in tqdm(range(self.cap_length)):
            img = self.cap.read()[1]
            for post in self.posts:
                post.reset
                post.update_transition(img)

            input = self.preprocess_frame(img)
            results = self.run_model(input)

            for confidence, (xn, yn, widthn, heightn) in zip(results['confidence'], results['coordinates']):
                if confidence > 0.3:
                    x, y, x1, y1, x2, y2 = get_coordinates(img, xn, yn, widthn, heightn)
                    for post in self.posts:
                        post.update_objects((x, y, x1, y1, x2, y2))

            current_time = frame_to_hms(self.cap.get(cv2.CAP_PROP_POS_FRAMES), self.cap_fps)
            for post in self.posts:
                post.forward(current_time)
                post.logger.update_logs()
        self.close_cap(current_time)


    def run_cam(self):
        try:
            while True:
                img = self.cap.read()[1]
                for post in self.posts:
                    post.reset
                    post.update_transition(img)

                input = self.preprocess_frame(img)
                results = self.run_model(input)

                for confidence, (xn, yn, widthn, heightn) in zip(results['confidence'], results['coordinates']):
                    if confidence > 0.3:
                        x, y, x1, y1, x2, y2 = get_coordinates(img, xn, yn, widthn, heightn)
                        for post in self.posts:
                            post.update_objects((x, y, x1, y1, x2, y2))

                current_time = frame_to_hms(self.cap.get(cv2.CAP_PROP_POS_FRAMES), self.cap_fps)
                for post in self.posts:
                    post.forward(current_time)
                    post.logger.update_logs()
        except KeyboardInterrupt:
            self.close_cap(current_time)
            exit()
        


#p = policy_plot('configs/benchmarkcam14_config', 'videos/benchmark_cam_14.mov')
#sp.run()





