
import sys

def search_pdf(file_path, search_terms):
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            
        found = {}
        for term in search_terms:
            # Search for the byte sequence of the term
            if term.encode('utf-8') in content:
                found[term] = True
            else:
                found[term] = False
                
        return found
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    pdf_path = "/Users/pawel/Documents/!Praca/Subverse/Human purchase intent/2510.08338v3.pdf"
    terms = [
        "github.com/pymc-labs/semantic-similarity-rating",
        "pymc-labs",
        "semantic-similarity-rating",
        "Human purchase intent" # Verify we have the right file/content
    ]
    
    results = search_pdf(pdf_path, terms)
    print(results)
