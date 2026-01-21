import markdown
from xhtml2pdf import pisa
import os

def convert_md_to_pdf(source_md, output_pdf):
    # 1. Read Markdown
    if not os.path.exists(source_md):
        print(f"Error: {source_md} not found.")
        return

    with open(source_md, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # 2. Convert to HTML
    # Use 'extra' extension for tables, fences, etc.
    html_content = markdown.markdown(md_text, extensions=['extra', 'codehilite'])

    # 3. Add Basic Styling for PDF
    full_html = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Helvetica, sans-serif; font-size: 12pt; line-height: 1.5; }}
        h1 {{ color: #2E3E50; border-bottom: 2px solid #2E3E50; padding-bottom: 5px; }}
        h2 {{ color: #E74C3C; margin-top: 20px; }}
        h3 {{ color: #3498DB; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """

    # 4. Write PDF
    with open(output_pdf, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

    if pisa_status.err:
        print(f"Error creating PDF for {source_md}")
    else:
        print(f"Successfully created: {output_pdf}")

if __name__ == "__main__":
    convert_md_to_pdf("research_proposal.md", "Research_Proposal.pdf")
    convert_md_to_pdf("README.md", "Project_Overview.pdf")
