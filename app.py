import streamlit as st
from langdetect import detect
from googletrans import Translator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np




# 👉 Đặt cấu hình trang ngay sau import
st.set_page_config(page_title="Metro Food Chatbot", layout="centered")

# 🌄 CSS ảnh nền toàn trang
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("https://cdn2.fptshop.com.vn/unsafe/800x0/hinh_nen_may_tinh_cute_hinh_4_658c54e9a2.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}}

[data-testid="stHeader"] {{
    background: rgba(255, 255, 255, 0);
}}

[data-testid="stSidebar"] {{
    background: rgba(255, 255, 255, 0.8);
}}

.block-container {{
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 10px;
    color: black;
}}

/* 👇 CSS màu chữ trong tin nhắn */
.user-message {{
    color: black !important;
    font-weight: normal;
}}

.assistant-message {{
    color: black !important;
    font-weight: normal;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)



# 🔁 Khởi tạo session state
if "food_model" not in st.session_state:
    try:
        st.session_state["food_model"] = load_model('food_recognition_model.h5')
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.session_state["food_model"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Xin chào! Hôm nay bạn muốn ăn gì? Gõ 'hết' nếu đặt đủ."}]
if "total_bill" not in st.session_state:
    st.session_state["total_bill"] = 0
if "detected_lang" not in st.session_state:
    st.session_state["detected_lang"] = None
if "ordering" not in st.session_state:
    st.session_state["ordering"] = False
if "current_dish" not in st.session_state:
    st.session_state["current_dish"] = None

# 🔠 Dữ liệu menu
dishes = ['bánh mì pate', 'bún chả', 'cơm tấm', 'phở bò', 'phở gà']
dishes_amount = {'bánh mì pate': 8, 'bún chả': 5, 'cơm tấm': 3, 'phở bò': 5, 'phở gà': 2}
menu = {'bánh mì pate': 20000, 'bún chả': 30000, 'cơm tấm': 25000, 'phở bò': 40000, 'phở gà': 35000}
dish_aliases = {
    'phở bò': ['phở bò', 'beef noodle'],
    'phở gà': ['phở gà', 'chicken noodle'],
    'bún chả': ['bún chả', 'grilled pork noodle'],
    'cơm tấm': ['cơm tấm', 'broken rice'],
    'bánh mì pate': ['bánh mì pate', 'pate bread']
}
end_keywords = ["hết", "done", "all", "complete", "終了", "完成"]

# 🌐 Dịch
translator = Translator()

# 👤 Avatar
USER_AVATAR ="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQqvbtyoWm5OyamfTuqTpOs-JYmt-Zq0WXJIw&s"
BOT_AVATAR ="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQb5qJdJPpSw5d0VMv1AD67Jc6_U_I9fBIhqQ&s"

# 🧠 Xử lý ảnh
def preprocess_uploaded_image(image):
    try:
        img = image.resize((64, 64))
        img_array = img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Lỗi xử lý ảnh: {e}")
        return None

def predict_food(processed_image):
    model = st.session_state.get("food_model")
    if model:
        try:
            predictions = model.predict(processed_image)
            idx = np.argmax(predictions)
            return dishes[idx], predictions[0][idx]
        except Exception as e:
            st.error(f"Lỗi dự đoán: {e}")
    return "Không thể nhận diện", 0.0

# 🧠 Xử lý ngôn ngữ và nội dung người dùng
def resolve_dish_name(user_input):
    for dish, aliases in dish_aliases.items():
        if user_input in aliases:
            return dish
    return None

def handle_user_input(prompt):
    lang = st.session_state["detected_lang"] or detect(prompt)
    st.session_state["detected_lang"] = lang
    question_vi = translator.translate(prompt, dest="vi").text.lower() if lang != "vi" else prompt.lower()

    response = ""
    ordering = st.session_state["ordering"]
    current_dish = st.session_state["current_dish"]

    if any(keyword in question_vi for keyword in ["mua", "đặt", "gọi", "thêm món"]) and not ordering:
        st.session_state["ordering"] = True
        response = "Bạn muốn đặt món gì? (gõ 'hết' nếu đặt đủ):"
    elif ordering:
        if question_vi in end_keywords:
            st.session_state["ordering"] = False
            response = f"Tổng tiền tạm tính: {st.session_state['total_bill']:,}đ."
        elif not current_dish:
            dish = resolve_dish_name(question_vi)
            if dish:
                st.session_state["current_dish"] = dish
                response = f"Bạn muốn bao nhiêu phần {dish}?"
            else:
                response = "Món này không có trong thực đơn."
        else:
            try:
                amount = int(question_vi)
                dish = current_dish
                if amount > dishes_amount[dish]:
                    response = f"Chỉ còn {dishes_amount[dish]} phần {dish}."
                else:
                    dishes_amount[dish] -= amount
                    total = menu[dish] * amount
                    st.session_state["total_bill"] += total
                    response = f"Đã thêm {amount} phần {dish} ({total:,}đ). Bạn muốn đặt thêm gì không?"
                    st.session_state["current_dish"] = None
            except ValueError:
                response = "Vui lòng nhập số lượng hợp lệ."
    elif "giờ mở cửa" in question_vi or "mấy giờ" in question_vi:
        response = "Chúng tôi mở cửa từ 6:00 đến 13:30 mỗi ngày."
    elif "món rẻ nhất" in question_vi:
        d = min(menu.items(), key=lambda x: x[1])
        response = f"Món rẻ nhất là {d[0]} ({d[1]:,}đ)."
    elif "món đắt nhất" in question_vi:
        d = max(menu.items(), key=lambda x: x[1])
        response = f"Món đắt nhất là {d[0]} ({d[1]:,}đ)."
    elif "menu" in question_vi or "thực đơn" in question_vi:
        response = "Thực đơn hôm nay:\n" + "\n".join(
            [f"- {d}: {menu[d]:,}đ (còn {dishes_amount[d]} phần)" for d in dishes]
        )
    elif "tổng tiền" in question_vi or "hóa đơn" in question_vi:
        response = f"Tổng hóa đơn: {st.session_state['total_bill']:,}đ."
    elif "giảm giá" in question_vi or "khuyến mãi" in question_vi:
        response = "Hiện chưa có chương trình khuyến mãi."
    elif "bán chạy nhất" in question_vi or "best seller" in question_vi:
        response = f"Món bán chạy nhất là {max(menu, key=menu.get)}."
    elif "xong" in question_vi and not ordering:
        response = f"Cảm ơn bạn! Tổng hóa đơn là {st.session_state['total_bill']:,}đ. Hẹn gặp lại!"
    else:
        response = "Xin lỗi, tôi chưa hiểu. Hãy thử hỏi về món ăn, thực đơn hoặc đặt món."

    translated = translator.translate(response, dest=lang).text if lang != "vi" else response
    st.session_state["messages"].append({"role": "assistant", "content": translated})
    st.chat_message("assistant", avatar=BOT_AVATAR).markdown(
        f'<div class="assistant-message">{translated}</div>', unsafe_allow_html=True)

# 📷 Nhận diện món từ ảnh
st.title("🍜 Metro Food Chatbot")
st.write("Bite the ride, taste the tide!")

uploaded_file = st.file_uploader("Tải ảnh món ăn", type=["jpg", "jpeg", "png"])
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh bạn đã tải", use_column_width=True)
        processed = preprocess_uploaded_image(image)
        if processed is not None:
            name, confidence = predict_food(processed)
            st.write(f"👉 Tôi dự đoán đây là **{name}** (độ tin cậy: {confidence:.2f})")
    except Exception as e:
        st.error(f"Lỗi ảnh: {e}")

# 🧾 Hiển thị đoạn chat
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user", avatar=USER_AVATAR).markdown(
            f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.chat_message("assistant", avatar=BOT_AVATAR).markdown(
            f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)


# 💬 Nhập văn bản
if prompt := st.chat_input("Bạn: "):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user", avatar=USER_AVATAR).write(prompt)
    handle_user_input(prompt)
