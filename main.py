import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import ( # pyright: ignore[reportMissingImports]
    MobileNetV2,
    decode_predictions
)
from PIL import Image
from image_utils import MAX_IMAGE_BYTES, load_uploaded_image, preprocess_image


def load_model():
    model = MobileNetV2(weights="imagenet")
    return model


def classify_image(model, image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return decode_predictions(predictions, top=3)[0]

def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")
    
    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what is in it!")
    
    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        if uploaded_file.size > MAX_IMAGE_BYTES:
            st.error("Image exceeds the 10 MB upload limit.")
            return
        try:
            image = load_uploaded_image(uploaded_file.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            return

        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Classify Image"):
            with st.spinner("Analyzing Image..."):
                try:
                    predictions = classify_image(model, image)
                except Exception:
                    st.error("Image classification failed. Check the model runtime and application logs.")
                    return
                st.subheader("Predictions")
                for _, label, score in predictions:
                    st.write(f"**{label}**: {score:.2%}")
                        
                        
if __name__ == "__main__":
    main()