from fastai.vision.all import *
import gradio as gr

learn = load_learner("car_random_truck.pkl")

categories = ('car','random','truck')
def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories,map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['car.jpg','truck.jpg','random.pjg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inlfine=False)

