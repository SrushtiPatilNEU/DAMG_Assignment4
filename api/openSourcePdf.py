import io
import fitz  # PyMuPDF
import base64
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
app = FastAPI()

def extract_data(pdf_file_io: BytesIO):
    try:
        doc = fitz.open(stream=pdf_file_io, filetype="pdf")
        
        text_data = ""
        tables = []
        images = []
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            text_data += f"### Page {page_num + 1}\n\n"
            text_data += page.get_text("text") + "\n\n"
            
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img = Image.open(BytesIO(image_bytes))
                img_filename = f"image_{page_num + 1}_{img_index + 1}.png"
                
                # Store image data directly instead of saving to file
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images.append({
                    "filename": img_filename,
                    "base64": img_base64
                })
            
            table = page.get_text("dict")
            tables.append(table)
        
        return {
            "text": text_data,
            "tables": tables,
            "images": images
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def save_to_md(extracted_data):
    try:
        markdown_content = "# Extracted Data from PDF\n\n"
        
        markdown_content += "## Extracted Text\n"
        markdown_content += extracted_data["text"]
        
        markdown_content += "## Extracted Tables\n"
        for table in extracted_data["tables"]:
            markdown_content += "### Table\n"
            for block in table["blocks"]:
                if block['type'] == 0:
                    for line in block["lines"]:
                        line_text = " | ".join([span["text"] for span in line["spans"]])
                        markdown_content += line_text + "\n"
            markdown_content += "\n"
        
        markdown_content += "## Extracted Images\n"
        for img in extracted_data["images"]:
            markdown_content += f"![{img['filename']}](data:image/png;base64,{img['base64']})\n"
        
        return markdown_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))