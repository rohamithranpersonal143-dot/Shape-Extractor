import streamlit as st
import cv2
import numpy as np
import math

st.set_page_config(page_title="Hand-Drawn Shape Analyzer", layout="wide")

st.title("📐 Hand-Drawn Shape Geometry Analyzer")
st.write("Upload an image of a shape drawn on paper to calculate its vertices, side lengths, and interior angles.")

def calculate_angle(p1, p2, p3):
    """Calculates the interior angle at vertex p2 between vectors p2->p1 and p2->p3."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
        
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = math.degrees(math.acos(cos_angle))
    return int(round(angle))

# Sidebar controls for fine-tuning
st.sidebar.header("🔧 Sensitivity Settings")
epsilon_slider = st.sidebar.slider("Hand Shakiness Tolerance (Epsilon %)", 1.0, 10.0, 4.0, 0.5) / 100.0
min_area = st.sidebar.slider("Minimum Shape Size Filter (Pixels)", 1000, 20000, 5000, 500)

uploaded_file = st.file_uploader("Choose an image of your drawn shape...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    display_frame = image.copy()

    # 1. Pre-process the frame to isolate dark pen lines on paper
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 2. Extract boundaries
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes_found = 0

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        # 3. Smooth out hand-drawn ripples to find the true structural corners
        epsilon = epsilon_slider * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        num_vertices = len(approx)
        if num_vertices >= 3:
            shapes_found += 1
            pts = [pt[0] for pt in approx]
            
            # Draw the clean geometric outline
            cv2.drawContours(display_frame, [approx], -1, (255, 0, 0), 3)

            # 4. Measure side lengths and corner angles
            for i in range(num_vertices):
                p_prev = pts[i - 1]
                p_curr = pts[i]
                p_next = pts[(i + 1) % num_vertices]
                
                side_len = np.linalg.norm(np.array(p_curr) - np.array(p_next))
                angle = calculate_angle(p_prev, p_curr, p_next)
                
                mid_x = int((p_curr[0] + p_next[0]) / 2)
                mid_y = int((p_curr[1] + p_next[1]) / 2)

                # Draw markers and data text
                cv2.circle(display_frame, tuple(p_curr), 8, (0, 255, 0), -1)
                cv2.putText(display_frame, f"{angle}°", (p_curr[0] + 10, p_curr[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, f"{int(side_len)}px", (mid_x, mid_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Display the total vertex count in the center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(display_frame, f"Vertices: {num_vertices}", (cX - 60, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)

    if shapes_found > 0:
        st.success(f"Successfully processed image! Detected {shapes_found} structural shape(s).")
        # Convert BGR (OpenCV layout) to RGB (Streamlit layout)
        st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.warning("No shapes detected. Try adjusting the sliders in the sidebar or upload a clearer picture with darker lines.")