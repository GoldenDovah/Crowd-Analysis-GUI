import csv
import time
from PIL import ImageTk
import PIL.Image
from tkinter import *
from customtkinter import *
import cv2
from config import YOLO_CONFIG, VIDEO_CONFIG, SHOW_PROCESSING_OUTPUT, DATA_RECORD_RATE, FRAME_SIZE, TRACK_MAX_AGE
from deep_sort import generate_detections as gdet, nn_matching
from deep_sort.tracker import Tracker
from video_process import video_process
from threading import Thread
from pygame import mixer
import os
from twilio.rest import Client


class App(CTk):

    
    def __init__(self):
        super().__init__()
        self.title('Human Behaviour Analysis')
        self.vid = self.thread_model = self.audio = None
        self.sms_sent = False
        self.call_sent = False
        self.var_stop = IntVar(value=0)
        self.var_alarm = IntVar(value=0)
        self.var_sms = IntVar(value=0)
        self.var_call = IntVar(value=0)
        self.var_abnormal = IntVar(value=0)
        self.var_abnormal.trace_add('write', self.abnormal_detected)
        self.img_alarm = ImageTk.PhotoImage(PIL.Image.open('alarm.png').resize((128,128)))
        self.img_sms = ImageTk.PhotoImage(PIL.Image.open('sms.png').resize((128,128)))
        self.img_call = ImageTk.PhotoImage(PIL.Image.open('call.png').resize((128,128)))
        if get_appearance_mode() == 'Dark':
            self.os_bg = '#333333'
            self.canvas_bg = 'gray12'
        else:
            self.os_bg = 'white'
            self.canvas_bg = 'gray2'
        mixer.init()
        self.screen_main()


    def abnormal_detected(self, *args):
        if self.var_abnormal.get():
            if self.var_alarm.get() and self.audio and not mixer.music.get_busy():
                mixer.music.play()
            if self.var_sms.get() and not self.sms_sent:
                self.sms_sent = True
                account_sid = "To Configure using TWILIO"
                auth_token = "To Configure using TWILIO"
                client = Client(account_sid, auth_token)
                message = client.messages.create(
                    body="Abnormal Behaviour Detected!",
                    from_="To Configure using TWILIO",
                    to="To Configure using TWILIO"
                )
                print(message.sid)
            if self.var_call.get() and not self.call_sent:
                self.call_sent = True
                account_sid = "To Configure using TWILIO"
                auth_token = "To Configure using TWILIO"
                client = Client(account_sid, auth_token)
                call = client.calls.create(
                    url='http://demo.twilio.com/docs/voice.xml',
                    to='To Configure using TWILIO',
                    from_='To Configure using TWILIO'
                )
                print(call.sid)
        elif mixer.music.get_busy():
            mixer.music.stop()


    def set_volume(self, value):
        mixer.music.set_volume(value/100)


    def screen_main(self):
        frm = CTkFrame(self)
        frm.pack(side='top', fill='both')
        self.canvas = Canvas(frm, width=1080, height=810, bd=0, bg=self.canvas_bg)
        self.canvas.pack(side='left')
        frm2 = CTkFrame(frm)
        frm2.pack(side='right', fill='both', expand=True)
        self.update_idletasks()
        Label(frm2, image=self.img_alarm, bg=self.os_bg).grid(row=0, column=0, padx=(20,5), pady=5)
        CTkSwitch(frm2, text="Alarm System", variable=self.var_alarm, onvalue=1, offvalue=0).grid(row=0, column=1, padx=(5,20), pady=5)
        self.button_audio = CTkButton(frm2, text='Select Audio', command=self.select_audio)
        self.button_audio.grid(row=1, column=0, columnspan=2, pady=5)
        CTkLabel(frm2, text='Volume').grid(row=2, column=0, padx=(20,5), pady=(5,50))
        CTkSlider(master=frm2, from_=0, to=100, command=self.set_volume).grid(row=2, column=1, padx=(5,20), pady=(5,50))
        Label(frm2, image=self.img_sms, bg=self.os_bg).grid(row=3, column=0, padx=(40,5), pady=(5,50))
        CTkSwitch(frm2, text="Get SMS", variable=self.var_sms, onvalue=1, offvalue=0).grid(row=3, column=1, padx=(5,20), pady=(5,50))
        Label(frm2, image=self.img_call, bg=self.os_bg).grid(row=4, column=0, padx=(0,5), pady=5)
        CTkSwitch(frm2, text="Get Call", variable=self.var_call, onvalue=1, offvalue=0).grid(row=4, column=1, padx=(5,20), pady=5)
        frm = CTkFrame(self)
        frm.pack(side='bottom', fill='x')
        frm.grid_columnconfigure((0,1), weight=1)
        CTkButton(frm, text='Select Video', command=self.select_video).grid(row=0, column=0, padx=15, pady=10)
        CTkButton(frm, text='Use Webcam', command=self.use_webcam).grid(row=0, column=1, padx=15, pady=10)


    def select_audio(self):
        filetypes = [
            ('Audio Files', 'mp3 wav')
        ]
        file_audio = filedialog.askopenfilename(filetypes=filetypes)
        if file_audio:
            self.sound = mixer.music.load(file_audio)
            self.button_audio.configure(text=os.path.basename(file_audio))
            self.audio = file_audio


    def use_webcam(self):
        if self.thread_model:
            print("HERE")
            self.var_stop.set(1)
            self.after(1000, self.delayed_cam)
        else:
            self.delayed_cam()
        
    def delayed_cam(self):
        self.var_stop.set(0)
        self.vid = cv2.VideoCapture(0)
        self.open_model()


    def open_model(self):
        IS_CAM = VIDEO_CONFIG["IS_CAM"]
        # Load YOLOv3-tiny weights and config
        WEIGHTS_PATH = YOLO_CONFIG["WEIGHTS_PATH"]
        CONFIG_PATH = YOLO_CONFIG["CONFIG_PATH"]
        # Load the YOLOv3-tiny pre-trained COCO dataset 
        net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
        # Set the preferable backend to CPU since we are not using GPU
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get the names of all the layers in the network
        ln = net.getLayerNames()
        # Filter out the layer names we dont need for YOLO
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        # Tracker parameters
        max_cosine_distance = 0.7
        nn_budget = None
        #initialize deep sort object
        if IS_CAM: 
            max_age = VIDEO_CONFIG["CAM_APPROX_FPS"] * TRACK_MAX_AGE
        else:
            max_age=DATA_RECORD_RATE * TRACK_MAX_AGE
            if max_age > 30:
                max_age = 30
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric, max_age=max_age)

        if not os.path.exists('processed_data'):
            os.makedirs('processed_data')
        movement_data_file = open('processed_data/movement_data.csv', 'w') 
        crowd_data_file = open('processed_data/crowd_data.csv', 'w')
        movement_data_writer = csv.writer(movement_data_file)
        crowd_data_writer = csv.writer(crowd_data_file)
        movement_data_writer.writerow(['Track ID', 'Entry time', 'Exit Time', 'Movement Tracks'])
        crowd_data_writer.writerow(['Time', 'Human Count', 'Social Distance violate', 'Restricted Entry', 'Abnormal Activity'])
        START_TIME = time.time()
        self.thread_model = Thread(target=video_process, args=(self.canvas, self.vid, FRAME_SIZE, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer, self.var_stop, self.var_abnormal))
        self.thread_model.start()
        #processing_FPS = video_process(self.canvas, self.vid, FRAME_SIZE, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer)


    def select_video(self):
        if self.thread_model:
            self.var_stop.set(1)
        filetypes = [
            ('Video Files', 'mp4 avi mkv')
        ]
        file_video = filedialog.askopenfilename(filetypes=filetypes)
        if file_video:
            self.vid = cv2.VideoCapture(file_video)
            self.var_stop.set(0)
            self.open_model()


if __name__ == '__main__':
    app = App()
    app.mainloop()
