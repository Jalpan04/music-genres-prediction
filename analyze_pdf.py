from pypdf import PdfReader
import sys

def extract_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        print(f"--- Analyzing: {pdf_path} ---")
        print(f"Total Pages: {len(reader.pages)}")
        
        # Read pages 8 to 11 (likely Result section)
        pages_to_read = [7, 8, 9, 10, 11]
        
        for i in pages_to_read:
            page = reader.pages[i]
            content = page.extract_text()
            print(f"\n--- Page {i+1} ---")
            print(content)
            
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    pdf_path = "A_Study_on_Music_Genre_Classification_using_Machin.pdf"
    extract_text(pdf_path)
