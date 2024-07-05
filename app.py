import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load your trained model
model = tf.keras.models.load_model('C:\Eye Disease Detection Project\eyes.h5')  # Adjust the path as needed

# Get the model's input shape
input_shape = model.input_shape[1:3]

# Define disease names for predictions
disease_names = {
    0: 'Cataract',
    1: 'Diabetic_retinopathy',
    2: 'Glaucoma',
    3: 'Normal'
}

# Function to preprocess the uploaded image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=input_shape)  # Resize to model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert single image to a batch
    img_array = img_array / 255.0  # Normalize the image array
    return img_array

# Function to make prediction
def predict(image_file):
    img_array = preprocess_image(image_file)
    prediction = model.predict(img_array)
    return prediction

# Function to get disease name from prediction
def get_disease_name(prediction):
    predicted_class = np.argmax(prediction, axis=1)[0]
    disease_name = disease_names.get(predicted_class, "Unknown Disease")
    return disease_name

# Sidebar content with additional features and information
def sidebar_content():
    st.sidebar.title("Welcome User")
    st.sidebar.markdown(
        """
        Welcome to the Eye Disease Detection Application. Upload an image of an eye to predict one of the following diseases:

        - **Cataract**
        - **Diabetic Retinopathy**
        - **Glaucoma**
        - **Normal**

        This application uses a deep learning model to analyze the image and provide a prediction.
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.header("About Eye Diseases")
    st.sidebar.markdown(
        """
        **Cataract**: A clouding of the lens in the eye.

        **Diabetic Retinopathy**: Damage to the retina due to diabetes.

        **Glaucoma**: Damage to the optic nerve often caused by elevated eye pressure.

        **Normal**: No detected abnormality in the eye.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Created by:")
    st.sidebar.subheader("Prajwal Gupta ❤️")
    st.sidebar.write("Email: 503prajwal@gmail.com")


# Streamlit app interface
def main():
    # Set page configurations
    st.set_page_config(
        page_title="Eye Disease Detection",
        page_icon=":eye:",
        layout="wide",
        initial_sidebar_state="expanded"  # Sidebar expanded by default
    )

    # Define custom CSS styles
    st.markdown(
        """
        <style>
        body {
            background-color: #F5F5DC;  /* Cream color background */
        }
        .sidebar .sidebar-content {
            background-color: #000000;  /* Black sidebar */
            color: white;  /* White text for sidebar */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Center-aligned main heading
    st.markdown("<h1 style='text-align: center;'>Eye Disease Detection</h1>", unsafe_allow_html=True)

    # Subheading 1: What are eye diseases? (Center-aligned)
    st.markdown("<h3 style='text-align: center;'>What are eye diseases?</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        Eye diseases include disorders that impact the structures directly surrounding your eyes as well as any part of your eye. These illnesses can be classified as acute, which means they manifest rapidly, or chronic, which manifests more slowly and lasts longer.
        """
    )

    # Subheading 2: How common are eye diseases?
    st.markdown("<h3 style='text-align: center;'>How common are eye diseases?</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        In general, eye diseases and vision disorders are extremely common. The World Health Organization estimates that over 2.2 billion people have some form of vision impairment or blindness.

        The fact that your eyes are a part of your body and do not exist in a vacuum is one reason why eye illnesses are so prevalent. Actually, the reverse is true. Many disorders affecting other body systems are the cause of, or related to, problems that affect your eyes. Because of this, there are several conditions that might harm your eyes.
        """
    )

    # Images with disease names
    st.markdown("<h3 style='text-align: center;'>Disease Predictor</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='display: flex; justify-content: center;'>
            <div style='text-align: center; padding: 10px;'>
                <img src='https://media.istockphoto.com/id/1356560292/photo/human-eye-with-clouded-lens-white-pupil-cataract-macro.jpg?s=612x612&w=0&k=20&c=Gvi0mtcNOtx5znzLUqEiHkrIGFMuMDNCNgkNU_B6_8c=' style='width: 100px; height: 100px; object-fit: cover; border-radius: 50%;'>
                <p>Cataract</p>
            </div>
            <div style='text-align: center; padding: 10px;'>
                <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJvNDqpVt5ELCtDQjrE7-Jey2F_j3KXAPixA&s' style='width: 100px; height: 100px; object-fit: cover; border-radius: 50%;'>
                <p>Diabetic Retinopathy</p>
            </div>
            <div style='text-align: center; padding: 10px;'>
                <img src='https://www.clinicasandiego.com.co/hs-fs/hubfs/glaucoma-3.webp?width=720&height=673&name=glaucoma-3.webp' style='width: 100px; height: 100px; object-fit: cover; border-radius: 50%;'>
                <p>Glaucoma</p>
            </div>
            <div style='text-align: center; padding: 10px;'>
                <img src='https://as1.ftcdn.net/v2/jpg/03/07/78/48/1000_F_307784863_82xzGcSHb1PVJEcP8kxyNzVJtAdZ9U48.jpg' style='width: 100px; height: 100px; object-fit: cover; border-radius: 50%;'>
                <p>Normal</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        """
        Upload an image of the eye to detect diseases.
        """
    )

    # File uploader with prediction buttons
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

            # Make prediction
            prediction = predict(uploaded_file)

            # Get disease name
            disease_name = get_disease_name(prediction)

            # Display the prediction result
            st.markdown("<h2 style='text-align: center; color: #71c33b;'>Predicted Disease</h2>",
                        unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-size: 24px;'>{}</p>".format(disease_name),
                        unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

    # Display sidebar content
    sidebar_content()


if __name__ == "__main__":
    main()
