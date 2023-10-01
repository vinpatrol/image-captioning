from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import streamlit as st


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gen_kwargs = {"max_length":20,"num_beams":4}

def generate_caption(image):
  if image.mode!="RGB":
    image = image.convert(mode="RGB")
  pixels = feature_extractor(images=image, return_tensors='pt').pixel_values
  pixels = pixels.to(device)

  output_ids = model.generate(pixels, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]

  return preds

#Streamlit UI

st.image("bionarybanner.png", use_column_width=False, output_format="PNG", width=600)

st.title("Image Captioning App")
st.write("Upload an image and get a caption!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        caption = generate_caption(image)
        # st.write("Predicted Caption:", caption)
        st.markdown(f"<h2 style='text-align: center; color: white;'>Predicted Caption:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 20px;'>{caption[0]}</p>", unsafe_allow_html=True)