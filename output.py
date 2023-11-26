import tensorflow as tf 
from utils import load_data, num_to_char
from model import get_model
import os


options = os.listdir(os.path.join('data', 's1'))
#print(options)

selected_video = 'bbaf2n.mpg'
file_path = os.path.join('data', 's1', selected_video)
#os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

video, annotations = load_data(tf.convert_to_tensor(file_path))

model = get_model()
print(f'model loaded')
yhat = model.predict(tf.expand_dims(video, axis=0))
decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
print(decoder)

converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
print(converted_prediction)