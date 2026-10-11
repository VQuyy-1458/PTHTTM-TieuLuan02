
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase


# =========================================================
# 1. CAU HINH DUONG DAN
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

STOCK_FILE = BASE_DIR / "data" / "stock" / "AAPL_stock_data.csv"

st.set_page_config(
    page_title="PTHTTM - Knowledge Graph & RAG",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# 2. DOC SECRETS
# =========================================================

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# =========================================================
# 3. NAP CAC TAI NGUYEN DA HUAN LUYEN
# =========================================================

@st.cache_resource
def load_resources():

    with open(
        OUTPUT_DIR / "rag_chunks.json",
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    vectorizer = joblib.load(
        OUTPUT_DIR / "rag_vectorizer.joblib"
    )

    matrix = load_npz(
        OUTPUT_DIR / "rag_tfidf_matrix.npz"
    )

    diabetes_model = joblib.load(
        MODEL_DIR / "diabetes_pipeline.joblib"
    )

    with open(
        MODEL_DIR / "diabetes_features.json",
        "r",
        encoding="utf-8"
    ) as f:
        feature_columns = json.load(f)

    with open(
        OUTPUT_DIR / "aapl_analysis.json",
        "r",
        encoding="utf-8"
    ) as f:
        stock_report = json.load(f)

    return (
        chunks,
        vectorizer,
        matrix,
        diabetes_model,
        feature_columns,
        stock_report
    )


# =========================================================
# 4. DOC VA LAM SACH DU LIEU CO PHIEU
# =========================================================

def normalize_column_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
    )


def find_column(columns, candidates):

    lookup = {
        normalize_column_name(col): col
        for col in columns
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)

        if key in lookup:
            return lookup[key]

    return None


@st.cache_data
def load_stock_data():

    if not STOCK_FILE.exists():
        raise FileNotFoundError(
            f"Khong tim thay file du lieu co phieu: {STOCK_FILE}"
        )

    df = pd.read_csv(STOCK_FILE)

    if df.empty:
        raise ValueError(
            "File du lieu co phieu khong co ban ghi."
        )

    date_col = find_column(
        df.columns,
        [
            "Date",
            "Datetime",
            "Timestamp",
            "Unnamed: 0"
        ]
    )

    close_col = find_column(
        df.columns,
        [
            "Close",
            "Adj Close",
            "Adjusted Close",
            "Adj_Close",
            "adjclose"
        ]
    )

    high_col = find_column(df.columns, ["High"])
    low_col = find_column(df.columns, ["Low"])
    volume_col = find_column(df.columns, ["Volume"])

    if date_col is None or close_col is None:
        raise ValueError(
            "Khong tim thay cot ngay hoac gia dong cua. "
            f"Cac cot hien co: {df.columns.tolist()}"
        )

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    for col in [close_col, high_col, low_col, volume_col]:
        if col is not None:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = (
        df.dropna(subset=[date_col, close_col])
        .sort_values(date_col)
        .drop_duplicates(subset=[date_col], keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "Khong con ban ghi co ngay va gia dong cua hop le. "
            "Hay kiem tra cot ngay trong file CSV."
        )

    ticker_col = find_column(
        df.columns,
        ["Ticker", "Symbol"]
    )

    if ticker_col is not None:
        ticker_values = df[ticker_col].astype(str).str.upper()
        aapl_rows = df[ticker_values == "AAPL"]

        if not aapl_rows.empty:
            df = aapl_rows.reset_index(drop=True)

    return {
        "df": df,
        "date_col": date_col,
        "close_col": close_col,
        "high_col": high_col,
        "low_col": low_col,
        "volume_col": volume_col
    }


# =========================================================
# 5. RAG - TRUY XUAT TAI LIEU
# =========================================================

def retrieve(question, category=None, top_k=3):

    chunks, vectorizer, matrix, *_ = load_resources()

    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector,
        matrix
    ).ravel()

    ranked = np.argsort(scores)[::-1]
    results = []

    for index in ranked:

        item = chunks[index]

        if category and item["category"] != category:
            continue

        results.append({
            "source": item["source"],
            "category": item["category"],
            "text": item["text"],
            "score": float(scores[index])
        })

        if len(results) >= top_k:
            break

    return results


def retrieval_answer(question, results):

    if not results:
        return (
            "Chua tim thay tai lieu phu hop. "
            "Hay thu dat cau hoi khac."
        )

    parts = [
        "Cau tra loi duoc tong hop tu cac doan tai lieu truy xuat:"
    ]

    for result in results:
        parts.append(f"- {result['text']}")

    parts.append(
        "Luu y: day la cau tra loi dua tren tai lieu truy xuat, "
        "chua phai noi dung do LLM sinh."
    )

    return "\n\n".join(parts)


# =========================================================
# 6. GEMINI - SINH CAU TRA LOI CHO RAG
# =========================================================

def generate_with_gemini(question, results):

    api_key = str(
        get_secret("GEMINI_API_KEY", "")
    ).strip()

    if not api_key:
        st.warning(
            "Chua tim thay GEMINI_API_KEY. "
            "Hay kiem tra file .streamlit/secrets.toml."
        )
        return retrieval_answer(question, results), False

    if not results:
        return (
            "Chua tim thay du tai lieu de tra loi.",
            False
        )

    try:
        from google import genai

        context = "\n\n".join(
            f"[Nguon: {item['source']}]\n{item['text']}"
            for item in results
        )

        client = genai.Client(api_key=api_key)

        prompt = f"""
Ban la tro ly hoc tap cho bai tap dai hoc ve Knowledge Graph va RAG.

Hay tra loi bang tieng Viet, ngan gon, de hieu.

Quy tac:
- Chi su dung thong tin trong ngu canh.
- Neu ngu canh khong du, hay noi ro chua du thong tin.
- Khong tu bia so lieu.
- Voi benh tieu duong, khong chan doan hay chi dinh dieu tri.
- Voi chung khoan, khong dua ra khuyen nghi mua hoac ban.
- Neu co the, ghi ro ten nguon duoc su dung.

NGU CANH:
{context}

CAU HOI:
{question}

Hay tra loi dua tren ngu canh.
"""

        response = None
        last_error = None

        # Thu toi da 3 lan neu Gemini bi qua tai 503.
        for attempt in range(3):

            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )

                # Goi API thanh cong thi dung vong lap.
                break

            except Exception as exc:

                error_text = str(exc).upper()

                is_overloaded = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                )

                # Khong thu lai neu la loi khac 503.
                if not is_overloaded:
                    raise

                last_error = exc

                # Neu con lan thu, cho roi thu lai.
                if attempt < 2:
                    wait_seconds = 2 ** (attempt + 1)

                    st.info(
                        f"Gemini dang qua tai. "
                        f"Thu lai sau {wait_seconds} giay "
                        f"(lan {attempt + 2}/3)."
                    )

                    time.sleep(wait_seconds)

        # Ca 3 lan deu bi 503.
        if response is None:

            if last_error is not None:
                raise last_error

            raise RuntimeError(
                "Gemini khong tra ve phan hoi."
            )

        answer = response.text

        if not answer or not answer.strip():
            raise RuntimeError(
                "Gemini khong tra ve noi dung van ban."
            )

        return answer.strip(), True

    except Exception as exc:

        error_type = type(exc).__name__
        error_message = str(exc)[:500]

        st.warning(
            "Gemini API chua hoat dong. "
            "He thong dang chuyen sang RAG truy xuat tai lieu."
        )

        st.caption(
            f"Chi tiet loi: {error_type}: {error_message}"
        )

        return retrieval_answer(question, results), False


# =========================================================
# 7. NEO4J - KNOWLEDGE GRAPH
# =========================================================

def query_neo4j(cypher, parameters=None):

    uri = get_secret("NEO4J_URI", "")
    username = get_secret("NEO4J_USERNAME", "")
    password = get_secret("NEO4J_PASSWORD", "")
    database = get_secret("NEO4J_DATABASE", "neo4j")

    if not uri or not username or not password:
        return None, (
            "Chua cau hinh Neo4j. "
            "Hay kiem tra muc Secrets."
        )

    driver = None

    try:
        driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )

        with driver.session(database=database) as session:
            records = session.run(
                cypher,
                parameters or {}
            )

            rows = [
                record.data()
                for record in records
            ]

        return pd.DataFrame(rows), None

    except Exception as exc:
        return None, (
            "Khong truy van duoc Neo4j. "
            f"Chi tiet: {str(exc)[:300]}"
        )

    finally:
        if driver is not None:
            driver.close()


# =========================================================
# 8. TIEU DE WEBSITE
# =========================================================

st.title("🧠 Hệ thống Knowledge Graph và RAG")

st.caption(
    "Tiểu luận 02 — Ứng dụng minh họa trên dữ liệu "
    "tiểu đường và cổ phiếu AAPL"
)


# =========================================================
# 9. NAP TAI NGUYEN
# =========================================================

try:

    (
        chunks,
        vectorizer,
        matrix,
        diabetes_model,
        feature_columns,
        stock_report
    ) = load_resources()

    st.success("Đã nạp tài nguyên hệ thống.")

except Exception as exc:

    st.error(
        "Không tải được tài nguyên. "
        "Hãy kiểm tra các file trong thư mục outputs/ và models/."
    )

    st.exception(exc)
    st.stop()


# =========================================================
# 10. TAO CAC TAB
# =========================================================

(
    overview_tab,
    stock_tab,
    rag_tab,
    diabetes_tab,
    graph_tab
) = st.tabs([
    "Tổng quan",
    "📈 Giá cổ phiếu AAPL",
    "Hỏi đáp RAG",
    "Dự đoán tiểu đường",
    "Knowledge Graph"
])


# =========================================================
# TAB 1 - TONG QUAN
# =========================================================

with overview_tab:

    st.subheader("Tổng quan hệ thống")

    col1, col2, col3 = st.columns(3)

    col1.metric("Số đoạn RAG", len(chunks))

    col2.metric(
        "Bản ghi AAPL",
        stock_report.get("rows", "N/A")
    )

    col3.metric(
        "Mã cổ phiếu",
        stock_report.get("ticker", "AAPL")
    )

    st.write(
        "Khoảng thời gian AAPL:",
        stock_report.get("start_date", "N/A"),
        "đến",
        stock_report.get("end_date", "N/A")
    )

    st.write(
        "Giá đóng cửa cuối cùng trong báo cáo:",
        stock_report.get("latest_close_in_dataset", "N/A")
    )

    chart_path = OUTPUT_DIR / "aapl_moving_averages.png"

    if chart_path.exists():
        st.image(
            str(chart_path),
            caption="AAPL — Giá đóng cửa và đường trung bình động",
            use_container_width=True
        )

    st.info(
        "Dữ liệu AAPL là dữ liệu lịch sử trong dataset, "
        "không phải giá thời gian thực."
    )


# =========================================================
# TAB 2 - PHAN TICH GIA CO PHIEU AAPL
# =========================================================

with stock_tab:

    st.header("Phân tích giá cổ phiếu Apple (AAPL)")

    st.write(
        "Tab này hiển thị dữ liệu lịch sử, biểu đồ giá đóng cửa "
        "và các chỉ số thống kê cơ bản của cổ phiếu AAPL."
    )

    try:

        stock_data = load_stock_data()

        stock_df = stock_data["df"]
        date_col = stock_data["date_col"]
        close_col = stock_data["close_col"]
        high_col = stock_data["high_col"]
        low_col = stock_data["low_col"]
        volume_col = stock_data["volume_col"]

        min_date = stock_df[date_col].min().date()
        max_date = stock_df[date_col].max().date()

        date_range = st.date_input(
            "Chọn khoảng thời gian phân tích",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="stock_date_range"
        )

        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

            start_date, end_date = date_range

            filtered_df = stock_df[
                (stock_df[date_col].dt.date >= start_date)
                & (stock_df[date_col].dt.date <= end_date)
            ].copy()

        else:
            filtered_df = stock_df.copy()

        if filtered_df.empty:

            st.warning(
                "Không có dữ liệu trong khoảng thời gian đã chọn."
            )

        else:

            filtered_df = filtered_df.sort_values(date_col)
            close_values = filtered_df[close_col].dropna()

            latest_close = float(close_values.iloc[-1])
            highest_close = float(close_values.max())
            lowest_close = float(close_values.min())

            if len(close_values) >= 2 and close_values.iloc[0] != 0:
                change_pct = (
                    (close_values.iloc[-1] - close_values.iloc[0])
                    / close_values.iloc[0]
                ) * 100
            else:
                change_pct = 0.0

            st.subheader(
                "Các chỉ số trong khoảng thời gian đã chọn"
            )

            metric1, metric2, metric3, metric4 = st.columns(4)

            metric1.metric(
                "Giá đóng cửa cuối kỳ",
                f"${latest_close:,.2f}"
            )

            metric2.metric(
                "Giá đóng cửa cao nhất",
                f"${highest_close:,.2f}"
            )

            metric3.metric(
                "Giá đóng cửa thấp nhất",
                f"${lowest_close:,.2f}"
            )

            metric4.metric(
                "Thay đổi trong kỳ",
                f"{change_pct:+.2f}%"
            )

            st.caption(
                "Các chỉ số được tính từ dữ liệu lịch sử trong file CSV. "
                "Giá và tỷ lệ thay đổi không phải dữ liệu thời gian thực."
            )

            # Bieu do gia dong cua
            st.subheader("Biểu đồ giá đóng cửa")

            price_chart = filtered_df[
                [date_col, close_col]
            ].copy()

            price_chart.columns = ["Date", "Close"]

            price_chart["Date"] = pd.to_datetime(
                price_chart["Date"],
                errors="coerce"
            )

            price_chart["Close"] = pd.to_numeric(
                price_chart["Close"],
                errors="coerce"
            )

            price_chart = (
                price_chart
                .dropna(subset=["Date", "Close"])
                .sort_values("Date")
                .reset_index(drop=True)
            )

            if not price_chart.empty:

                line_data = price_chart.set_index("Date")[["Close"]]

                st.line_chart(
                    line_data,
                    y="Close",
                    x_label="Ngày",
                    y_label="Giá (USD)"
                )

            else:
                st.warning(
                    "Không có dữ liệu hợp lệ để vẽ biểu đồ giá."
                )

            # Bieu do khoi luong giao dich
            if volume_col is not None:

                st.subheader("Khối lượng giao dịch")

                volume_chart = filtered_df[
                    [date_col, volume_col]
                ].copy()

                volume_chart.columns = ["Date", "Volume"]

                volume_chart["Date"] = pd.to_datetime(
                    volume_chart["Date"],
                    errors="coerce"
                )

                volume_chart["Volume"] = pd.to_numeric(
                    volume_chart["Volume"],
                    errors="coerce"
                )

                volume_chart = (
                    volume_chart
                    .dropna(subset=["Date", "Volume"])
                    .sort_values("Date")
                    .reset_index(drop=True)
                )

                if not volume_chart.empty:

                    bar_data = volume_chart.set_index("Date")[["Volume"]]

                    st.bar_chart(
                        bar_data,
                        y="Volume",
                        x_label="Ngày",
                        y_label="Khối lượng"
                    )

                else:
                    st.info(
                        "Không có dữ liệu khối lượng giao dịch hợp lệ."
                    )

            # Bang du lieu lich su
            st.subheader("Bảng dữ liệu cổ phiếu")

            display_df = filtered_df.copy()

            display_df = display_df.rename(
                columns={
                    date_col: "Date",
                    close_col: "Close"
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                label="Tải dữ liệu CSV đã lọc",
                data=display_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="AAPL_filtered_data.csv",
                mime="text/csv"
            )

            with st.expander("Thông tin dataset"):

                st.write("**File dữ liệu:**", str(STOCK_FILE))
                st.write("**Số bản ghi sau khi lọc:**", len(filtered_df))
                st.write("**Cột ngày:**", date_col)
                st.write("**Cột giá đóng cửa:**", close_col)
                st.write("**Cột giá cao nhất:**", high_col or "Không có")
                st.write("**Cột giá thấp nhất:**", low_col or "Không có")
                st.write("**Cột khối lượng:**", volume_col or "Không có")

    except FileNotFoundError:

        st.error("Không tìm thấy file dữ liệu cổ phiếu.")
        st.code(str(STOCK_FILE))

        st.write(
            "Hãy đặt file AAPL_stock_data.csv vào thư mục "
            "data/stock/ của dự án."
        )

    except Exception as exc:

        st.error("Không thể phân tích dữ liệu cổ phiếu.")
        st.exception(exc)


# =========================================================
# TAB 3 - RAG
# =========================================================

with rag_tab:

    st.subheader("Hỏi đáp trên Knowledge Base")

    category_label = st.selectbox(
        "Chọn lĩnh vực",
        [
            "Tất cả",
            "Bệnh tiểu đường",
            "Chứng khoán AAPL"
        ]
    )

    category_map = {
        "Tất cả": None,
        "Bệnh tiểu đường": "diabetes",
        "Chứng khoán AAPL": "stock"
    }

    question = st.text_area(
        "Nhập câu hỏi",
        placeholder=(
            "Ví dụ: Dataset AAPL có những trường dữ liệu nào?"
        )
    )

    if st.button(
        "Tìm tài liệu và trả lời",
        key="rag_button"
    ):

        if not question.strip():

            st.warning("Hãy nhập câu hỏi trước.")

        else:

            try:

                with st.spinner(
                    "Đang truy xuất tài liệu và tạo câu trả lời..."
                ):

                    results = retrieve(
                        question,
                        category=category_map[category_label],
                        top_k=3
                    )

                    answer, used_llm = generate_with_gemini(
                        question,
                        results
                    )

                st.markdown("### Câu trả lời")
                st.write(answer)

                if used_llm:

                    st.success(
                        "Câu trả lời được sinh bằng Gemini "
                        "dựa trên ngữ cảnh truy xuất."
                    )

                else:

                    st.caption(
                        "Đang dùng chế độ truy xuất tài liệu "
                        "vì LLM chưa được cấu hình hoặc API không khả dụng."
                    )

                st.markdown("### Nguồn truy xuất")

                if not results:
                    st.info("Không tìm thấy nguồn phù hợp.")

                for item in results:

                    with st.expander(
                        f"{item['source']} — "
                        f"độ tương đồng {item['score']:.3f}"
                    ):
                        st.write(item["text"])

            except Exception as exc:

                st.error("Không thể thực hiện truy vấn RAG.")
                st.exception(exc)


# =========================================================
# TAB 4 - DU DOAN TIEU DUONG
# =========================================================

with diabetes_tab:

    st.subheader("Dự đoán minh họa bệnh tiểu đường")

    st.warning(
        "Công cụ học tập, không phải thiết bị y tế "
        "và không dùng để chẩn đoán."
    )

    with st.form("diabetes_form"):

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=30,
            value=2
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=400.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "BloodPressure",
            min_value=0.0,
            max_value=250.0,
            value=70.0
        )

        skin_thickness = st.number_input(
            "SkinThickness",
            min_value=0.0,
            max_value=150.0,
            value=25.0
        )

        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=1000.0,
            value=80.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=27.5
        )

        pedigree = st.number_input(
            "DiabetesPedigreeFunction",
            min_value=0.0,
            max_value=5.0,
            value=0.4
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=35
        )

        submitted = st.form_submit_button(
            "Thực hiện dự đoán"
        )

    if submitted:

        try:

            sample = pd.DataFrame([{
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": pedigree,
                "Age": age
            }])[feature_columns]

            prediction = int(
                diabetes_model.predict(sample)[0]
            )

            probability = float(
                diabetes_model.predict_proba(sample)[0, 1]
            )

            st.metric("Lớp dự đoán", prediction)

            st.metric(
                "Xác suất mô hình gán cho lớp 1",
                f"{probability:.1%}"
            )

            st.caption(
                "Kết quả do mô hình học máy tạo ra trên dữ liệu đầu vào; "
                "không phải xác suất chẩn đoán lâm sàng."
            )

        except Exception as exc:

            st.error("Không thể thực hiện dự đoán.")
            st.exception(exc)


# =========================================================
# TAB 5 - KNOWLEDGE GRAPH
# =========================================================

with graph_tab:

    st.subheader("Truy vấn Knowledge Graph từ Neo4j")

    graph_choice = st.selectbox(
        "Chọn truy vấn",
        [
            "Toàn bộ quan hệ",
            "Bộ dữ liệu tiểu đường",
            "Cổ phiếu AAPL"
        ]
    )

    if graph_choice == "Bộ dữ liệu tiểu đường":

        cypher = """
        MATCH (a)-[r]->(b)
        WHERE a.name CONTAINS 'Pima'
           OR b.name IN [
                'Glucose', 'BMI', 'BloodPressure',
                'Age', 'Outcome'
           ]
        RETURN
            a.name AS source,
            type(r) AS relation,
            b.name AS target
        ORDER BY source
        """

    elif graph_choice == "Cổ phiếu AAPL":

        cypher = """
        MATCH (a)-[r]->(b)
        WHERE a.name IN [
            'Apple Inc.', 'AAPL', 'AAPL Stock History'
        ]
        RETURN
            a.name AS source,
            type(r) AS relation,
            b.name AS target
        ORDER BY source
        """

    else:

        cypher = """
        MATCH (a)-[r]->(b)
        RETURN
            a.name AS source,
            type(r) AS relation,
            b.name AS target
        ORDER BY source
        """

    if st.button(
        "Truy vấn Neo4j",
        key="graph_button"
    ):

        graph_df, error = query_neo4j(cypher)

        if error:

            st.error(error)

        elif graph_df is not None and graph_df.empty:

            st.info(
                "Không tìm thấy quan hệ. "
                "Hãy chạy cell tạo Knowledge Graph trong notebook."
            )

        else:

            st.dataframe(
                graph_df,
                use_container_width=True
            )

            st.success(
                f"Đã truy vấn {len(graph_df)} quan hệ từ Neo4j."
            )


# =========================================================
# CHAN TRANG
# =========================================================

st.divider()

st.caption(
    "PTHTTM — Tiểu luận 02. "
    "Hệ thống phục vụ học tập và minh họa công nghệ AI."
)
