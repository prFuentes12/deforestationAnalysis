import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageSequence
from docx import Document

def extract_gif_from_doc(doc_path, output_folder):
    """Extrae un GIF incrustado en un documento de Word."""
    doc = Document(doc_path)
    os.makedirs(output_folder, exist_ok=True)
    img_filename = os.path.join(output_folder, "gif_extraido.gif")

    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            img_data = rel.target_part.blob
            with open(img_filename, "wb") as f:
                f.write(img_data)
            print(f"GIF extraído: {img_filename}")
            return img_filename
    return None

def extract_frames_from_gif(gif_path, output_folder, years):
    """Extrae los fotogramas del GIF y los guarda con nombres correspondientes a los años en orden."""
    with Image.open(gif_path) as gif:
        for i, frame in enumerate(ImageSequence.Iterator(gif)):
            if i < len(years):
                frame_path = os.path.join(output_folder, f"{years[i]}.png")
                frame.save(frame_path)
                print(f"Fotograma {years[i]} guardado en: {frame_path}")

def preprocess_image(image_path):
    """Preprocesa la imagen: convierte a escala de grises, aplica filtro de mediana y eliminación de nubes."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.medianBlur(gray, 5)

    # Aplicar detección de nubes y eliminar áreas blancas
    _, cloud_mask = cv2.threshold(filtered, 200, 255, cv2.THRESH_BINARY)
    cleaned = cv2.inpaint(filtered, cloud_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    normalized = cv2.normalize(cleaned, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def segment_deforestation(image):
    """Segmenta las áreas deforestadas utilizando umbral adaptativo y técnicas morfológicas para limpieza."""
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 5)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    return cleaned

def calculate_deforested_area(binary_image, pixel_to_km_ratio=(20/51)**2):
    """Calcula el área deforestada en km² basada en la segmentación de la imagen."""
    white_pixels = np.sum(binary_image == 0)  # Contar píxeles negros (área segmentada)
    deforested_area = white_pixels * pixel_to_km_ratio
    return deforested_area

def main():
    doc_path = "deforestacion.docx"
    output_folder = "frames_extraidos"

    years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
             2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    gif_path = extract_gif_from_doc(doc_path, output_folder)
    if gif_path:
        extract_frames_from_gif(gif_path, output_folder, years)

    image_files = sorted([os.path.join(output_folder, f"{year}.png") 
                          for year in years if os.path.exists(os.path.join(output_folder, f"{year}.png"))])

    deforested_areas = {}

    for image_path, year in zip(image_files, years):
        processed = preprocess_image(image_path)
        segmented = segment_deforestation(processed)
        area = calculate_deforested_area(segmented)

        deforested_areas[year] = area

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.imread(image_path))
        plt.title(f'Imagen Original - {year}')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(processed, cmap='gray')
        plt.title(f'Preprocesada - {year}')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(segmented, cmap='gray')
        plt.title(f'Segmentada - {year} (Área: {area:.2f} km²)')
        plt.axis('off')

        plt.show()

    # Crear gráfica de evolución de la deforestación
    plt.figure(figsize=(10, 5))
    plt.plot(years[:len(deforested_areas)], list(deforested_areas.values()), 
             marker='o', linestyle='-', color='green')
    plt.xlabel("Año")
    plt.ylabel("Área Deforestada (km²)")
    plt.title("Evolución de la Deforestación")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
