"""
This program is an image labeling tool designed for annotating images with bounding boxes.
It supports both manual and automatic labeling modes. Users can draw rectangles around objects
in the images and assign class labels to them. The tool saves the annotations in both XML and YOLO
formats. It also allows users to load existing annotations, adjust the threshold for automatic
labeling, and generate YOLO files from XML annotations in a selected folder.
"""

import os
import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import QApplication, QMainWindow, QStatusBar, QAction, QFileDialog, QLabel, QVBoxLayout, QWidget, QInputDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QRect, QSize
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rectangles = []
        self.auto_label_mode = False
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.original_pixmap = None
        self.parent = parent
        self.image = None
        self.setFixedWidth(1024)  # Set fixed width to 1024 pixels
        self.setStyleSheet("background-color: green;")
        self.setMouseTracking(True)  # Enable mouse tracking
        self.class_char = None
        self.class_labels = {}
        self.current_rect = None
        self.changes_made = False  # Flag to track changes

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.adjustHeightToPixmap()
        scaled_pixmap = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(scaled_pixmap)

    def adjustHeightToPixmap(self):
        if self.original_pixmap:
            pixmap_width = self.original_pixmap.width()
            pixmap_height = self.original_pixmap.height()
            aspect_ratio = pixmap_height / pixmap_width
            new_width = 1024  # Fixed width
            new_height = int(new_width * aspect_ratio)
            self.setFixedSize(new_width, new_height)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.original_pixmap:
            painter = QPainter(self)
            for rect, color, class_char in self.rectangles:
                screen_rect = self.map_to_screen_coordinates(rect)
                pen = QPen(color, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(screen_rect)
                if class_char:
                    painter.setPen(QPen(Qt.blue))  # Change color to blue
                    painter.setFont(QFont("SimHei", 20))  # Set font to SimHei for Chinese characters
                    text_rect = painter.boundingRect(screen_rect, Qt.AlignCenter, class_char)
                    painter.drawText(text_rect, Qt.AlignCenter, class_char)
            if self.drawing and self.start_point and self.end_point:
                pen = QPen(Qt.red, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(QRect(self.start_point, self.end_point))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.pixmap() is None:
                return
            if self.parent.auto_label_mode:
                return
                # self.start_point = event.pos()
                # self.detect_white_area(self.start_point)
                # self.drawing = False
            else:
                self.start_point = event.pos()
                self.end_point = event.pos()
                self.drawing = True
                self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()
        else:
            self.update_mouse_position(event.pos())
            self.check_mouse_in_rectangle(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_point = event.pos()
            self.drawing = False
            if not self.parent.auto_label_mode:
                offset_x = (self.original_pixmap.width() - self.width()) / 2
                offset_y = (self.original_pixmap.height() - self.height()) / 2
                rect = QRect(
                    int(self.start_point.x() + offset_x),
                    int(self.start_point.y() + offset_y),
                    int((self.end_point.x() - self.start_point.x())),
                    int((self.end_point.y() - self.start_point.y()))
                )
            else:
                scale_x = self.original_pixmap.width() / self.width()
                scale_y = self.original_pixmap.height() / self.height()
                rect = QRect(
                    int(self.start_point.x() * scale_x),
                    int(self.start_point.y() * scale_y),
                    int((self.end_point.x() - self.start_point.x()) * scale_x),
                    int((self.end_point.y() - self.start_point.y()) * scale_y)
                )
            self.add_rectangle(rect)
            self.update()

    def keyPressEvent(self, event):
        key = event.text().lower()
        if key in self.class_labels and self.current_rect:
            self.class_char = self.class_labels[key]
            for i, (rect, color, _) in enumerate(self.rectangles):
                if rect == self.current_rect:
                    self.rectangles[i] = (rect, color, self.class_char)
                    self.changes_made = True  # Mark changes made
                    self.update()
                    break

    def add_rectangle(self, rect):
        color = Qt.red  # Default color, will be updated based on class
        self.rectangles.append((rect, color, self.class_char))
        self.changes_made = True  # Mark changes made
        self.update()
        self.parent.update_remove_all_rectangles_action()

    def update_last_rectangle_color(self, class_name):
        color = self.get_color_for_class(class_name)
        self.rectangles[-1] = (self.rectangles[-1][0], color, self.rectangles[-1][2])
        self.update()

    def get_color_for_class(self, class_name):
        # Generate a color based on the class name
        hash_code = hash(class_name)
        r = (hash_code & 0xFF0000) >> 16
        g = (hash_code & 0x00FF00) >> 8
        b = hash_code & 0x0000FF
        return QColor(r % 256, g % 256, b % 256)

    def save_labels(self, image_path, threshold):
        if not self.rectangles or not self.changes_made:
            return  # Do not save labels if there are no rectangles or no changes made
        label_path = os.path.splitext(image_path)[0] + ".xml"
        yolo_label_path = os.path.splitext(image_path)[0] + ".txt"
        root = ET.Element("annotation")

        folder = ET.SubElement(root, "folder")
        folder.text = os.path.basename(os.path.dirname(image_path))

        filename = ET.SubElement(root, "filename")
        filename.text = os.path.basename(image_path)

        path = ET.SubElement(root, "path")
        path.text = image_path

        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"

        # Get the actual image dimensions from the file on disk
        jpeg_height, jpeg_width = self.parent.current_grayscale_image.shape

        size = ET.SubElement(root, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = str(jpeg_width)
        height_elem = ET.SubElement(size, "height")
        height_elem.text = str(jpeg_height)
        depth = ET.SubElement(size, "depth")
        depth.text = "3"  # Assuming RGB images

        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"

        threshold_elem = ET.SubElement(root, "threshold")
        threshold_elem.text = str(threshold)

        yolo_lines = []

        for rect, color, class_char in self.rectangles:
            obj = ET.SubElement(root, "object")
            name = ET.SubElement(obj, "name")
            name.text = class_char  # Use the class character
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            
            # Adjust coordinates to match the original image dimensions
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(rect.left() / self.width())
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(rect.top() / self.height())
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(rect.right() / self.width())
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(rect.bottom() / self.height())

            # Add view coordinates
            xmin_view = ET.SubElement(bndbox, "xmin_view")
            xmin_view.text = str(rect.left() / self.width())
            ymin_view = ET.SubElement(bndbox, "ymin_view")
            ymin_view.text = str(rect.top() / self.height())
            xmax_view = ET.SubElement(bndbox, "xmax_view")
            xmax_view.text = str(rect.right() / self.width())
            ymax_view = ET.SubElement(bndbox, "ymax_view")
            ymax_view.text = str(rect.bottom() / self.height())

            # YOLO format: class_index x_center y_center width height
            class_index = list(self.class_labels.values()).index(class_char)
            x_center = (rect.left() + rect.width() / 2) / self.width()
            y_center = (rect.top() + rect.height() / 2) / self.height()
            bbox_width = rect.width() / self.width()
            bbox_height = rect.height() / self.height()
            yolo_line = f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}"
            yolo_lines.append(yolo_line)

            # Debug print statements
            print(f"Saving rect: {rect}, view coordinates: ({rect.left() / self.width()}, {rect.top() / self.height()}, {rect.right() / self.width()}, {rect.bottom() / self.height()})")

        tree = ET.ElementTree(root)
        tree.write(label_path, encoding='utf-8', xml_declaration=True)
        self.indent_xml(label_path)
        self.changes_made = False  # Reset changes made flag

        # Save YOLO-format label file
        with open(yolo_label_path, 'w') as yolo_file:
            yolo_file.write("\n".join(yolo_lines))

    def indent_xml(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        self._indent_element(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)

    def _indent_element(self, elem, level=0):
        i = "\n" + level * "    "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "    "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent_element(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def load_labels_from_xml(self, image_path):
        label_path = os.path.splitext(image_path)[0] + ".xml"
        self.rectangles = []  # Clear existing rectangles
        if not os.path.exists(label_path):
            self.update()
            return None
        try:
            tree = ET.parse(label_path)
        except ET.ParseError as e:
            print(f"Error parsing XML file {label_path}: {e}")
            return None
        root = tree.getroot()
        threshold_elem = root.find("threshold")
        threshold = int(threshold_elem.text) if threshold_elem is not None else None

        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin_view = float(bndbox.find("xmin_view").text) * self.width()
            ymin_view = float(bndbox.find("ymin_view").text) * self.height()
            xmax_view = float(bndbox.find("xmax_view").text) * self.width()
            ymax_view = float(bndbox.find("ymax_view").text) * self.height()
            
            rect = QRect(
                int(round(xmin_view)),
                int(round(ymin_view)),
                int(round(xmax_view - xmin_view)),
                int(round(ymax_view - ymin_view))
            )
            class_char = obj.find("name").text
            color = self.get_color_for_class(class_char)  # Update color based on class
            self.rectangles.append((rect, color, class_char))

            # Debug print statements
            print(f"Loading rect: {rect}, view coordinates: ({xmin_view}, {ymin_view}, {xmax_view}, {ymax_view})")

        self.update()
        return threshold

    def map_to_pixmap_coordinates(self, point):
        if self.pixmap() is None:
            return None, None  # Return None if there is no pixmap

        # Calculate the scale factors and offsets
        label_width = self.width()
        label_height = self.height()
        pixmap_width = self.pixmap().width()
        pixmap_height = self.pixmap().height()

        # Calculate the aspect ratios
        label_aspect_ratio = label_width / label_height
        pixmap_aspect_ratio = pixmap_width / pixmap_height

        if label_aspect_ratio > pixmap_aspect_ratio:
            # Label is wider than pixmap
            scale = label_height / pixmap_height
            offset_x = (label_width - pixmap_width * scale) / 2
            offset_y = 0
        else:
            # Label is taller than pixmap
            scale = label_width / pixmap_width
            offset_x = 0
            offset_y = (label_height - pixmap_height * scale) / 2

        # Calculate the coordinates in the original pixmap
        x = int(round(((point.x() - offset_x) / scale)))
        y = int(round((point.y() - offset_y) / scale))

        return x, y

    def map_to_screen_coordinates(self, rect):
        # Calculate the scale factors and offsets
        label_width = self.width()
        label_height = self.height()
        pixmap_width = self.original_pixmap.width()
        pixmap_height = self.original_pixmap.height()

        # Calculate the aspect ratios
        label_aspect_ratio = label_width / label_height
        pixmap_aspect_ratio = pixmap_width / pixmap_height

        if label_aspect_ratio > pixmap_aspect_ratio:
            # Label is wider than pixmap
            scale = label_height / pixmap_height
            offset_x = (label_width - pixmap_width * scale) / 2
            offset_y = 0
        else:
            # Label is taller than pixmap
            scale = label_width / pixmap_width
            offset_x = 0
            offset_y = (label_height - pixmap_height * scale) / 2

        # Calculate the coordinates in the QLabel
        x = int(rect.left() * scale + offset_x)
        y = int(rect.top() * scale + offset_y)
        width = int(rect.width() * scale)
        height = int(rect.height() * scale)

        return QRect(x, y, width, height)

    def detect_white_area(self, point):
        if self.pixmap() is None:
            return

        # Convert the point to the coordinates of the original pixmap
        x, y = self.map_to_pixmap_coordinates(point)
        if x is None or y is None or x < 0 or y < 0:
            return
        if x >= self.pixmap().width() or y >= self.pixmap().height():
            return

        # Convert the image to a format suitable for processing
        image_array = self.pixmap_to_binary_array(self.pixmap())

        # Check if the clicked point is white
        target_pixel = image_array[y, x]
        if not self.is_white(target_pixel):
            return

        # Use flood fill to detect the white area
        mask = np.zeros((image_array.shape[0] + 2, image_array.shape[1] + 2), np.uint8)
        cv2.floodFill(image_array, mask, (x, y), 255)

        # Find the bounding box of the white area
        white_pixels = np.where(mask[1:-1, 1:-1] == 1)
        min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
        min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])

        # Enhance the rectangle by 2 pixels in each direction
        min_x = max(min_x - 2, 0)
        max_x = min(max_x + 2, image_array.shape[1] - 1)
        min_y = max(min_y - 4, 0)
        max_y = min(max_y + 4, image_array.shape[0] - 1)

        # Get the actual image dimensions
        jpeg_height, jpeg_width = self.parent.current_grayscale_image.shape

        # Scale down to the original image size and round to the nearest integer
        scale_x = self.width() / jpeg_width
        scale_y = self.height() / jpeg_height
        min_x = round(min_x / scale_x)
        max_x = round(max_x / scale_x)
        min_y = round(min_y / scale_y)
        max_y = round(max_y / scale_y)

        # Scale up to the pixmap size
        min_x = int(min_x * scale_x)
        max_x = int(max_x * scale_x)
        min_y = int(min_y * scale_y)
        max_y = int(max_y * scale_y)

        rect = QRect(min_x, min_y, max_x - min_x, max_y - min_y)
        self.add_rectangle(rect)
        self.update()

    def is_white(self, pixel):
        return (pixel > 128)

    def pixmap_to_binary_array(self, pixmap, threshold=128):
        # Convert QPixmap to QImage
        image = pixmap.toImage()
        
        # Convert QImage to grayscale
        grayscale_image = image.convertToFormat(QImage.Format_Grayscale8)
        
        # Extract pixel data from QImage
        width = grayscale_image.width()
        height = grayscale_image.height()
        stride = grayscale_image.bytesPerLine()
        ptr = grayscale_image.bits()
        ptr.setsize(height * stride)
        
        # Convert to 2D numpy array
        array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, stride))
        
        # Crop the array to the actual width of the image
        array = array[:, :width]
    
        # Apply threshold to convert to binary image
        binary_array = np.where(array > threshold, 255, 0).astype(np.uint8)
        
        # Save binary image to disk in JPEG format
        binary_image_path = "binary_image.jpg"
        cv2.imwrite(binary_image_path, binary_array)
        
        return binary_array

    def update_mouse_position(self, pos):
        self.start_point = pos
        #self.parent.setWindowTitle(f"Image Labeling Tool     ({pos.x()}, {pos.y()})")

    def resizeEvent(self, event):
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(scaled_pixmap)
        super().resizeEvent(event)

    def check_mouse_in_rectangle(self, pos):
        if self.original_pixmap is None:
            self.current_rect = None
            return
        scale_x = self.original_pixmap.width() / self.width()
        scale_y = self.original_pixmap.height() / self.height()
        for rect, color, class_char in self.rectangles:
            screen_rect = QRect(
                int(rect.left() / scale_x),
                int(rect.top() / scale_y),
                int(rect.width() / scale_x),
                int(rect.height() / scale_y)
            )
            if screen_rect.contains(pos):
                self.current_rect = rect
                return
        self.current_rect = None

    def remove_all_rectangles(self):
        self.rectangles.clear()
        self.update()
        filename = self.parent.image_files[self.parent.current_image_index]
        filename = filename.replace(".jpg", ".xml")
        if os.path.exists(filename):
            os.remove(filename)
        self.parent.update_remove_all_rectangles_action()

class ImageLabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Labeling Tool")
        self.setGeometry(100, 100, 1000, 500)
        
        self.menu = self.menuBar()
        
        # Interchange the position of the "Image" and "Class" menus
        self.class_menu = self.menu.addMenu("Class")
        load_class_labels_action = QAction("Load Class Labels", self)
        load_class_labels_action.triggered.connect(self.load_class_labels)
        load_class_labels_action.setShortcut('Alt+C')  # Add shortcut key 'Alt+C'
        self.class_menu.addAction(load_class_labels_action)

        self.remove_all_rectangles_action = QAction("Remove All Rectangles", self)
        self.remove_all_rectangles_action.setEnabled(False)  # Initially disabled
        self.class_menu.addAction(self.remove_all_rectangles_action)

        self.file_menu = self.menu.addMenu("Image")
        self.open_folder_action = QAction("Open Folder", self)
        self.open_folder_action.setEnabled(False)  # Disable initially
        self.open_folder_action.triggered.connect(self.open_folder)
        self.open_folder_action.setShortcut('Alt+F')  # Add shortcut key 'Alt+F'
        self.file_menu.addAction(self.open_folder_action)
        
        self.save_xml_action = QAction("Save XML", self)
        self.save_xml_action.setEnabled(False)  # Disable initially
        self.save_xml_action.triggered.connect(self.save_xml)
        self.file_menu.addAction(self.save_xml_action)
        
        self.load_xml_action = QAction("Load XML", self)
        self.load_xml_action.setEnabled(False)  # Disable initially
        self.load_xml_action.triggered.connect(self.load_xml)
        self.file_menu.addAction(self.load_xml_action)

        self.generate_yolo_action = QAction("Generate YOLO Files", self)
        self.generate_yolo_action.setEnabled(False)  # Initially disabled
        self.generate_yolo_action.triggered.connect(self.generate_yolo_files)
        self.file_menu.addAction(self.generate_yolo_action)
        
        self.mouse_position_layout = QVBoxLayout()
        
        self.canvas = ImageLabel(self)
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setCursor(Qt.CrossCursor)
        
        self.mouse_position_layout.addWidget(self.canvas)
        
        container = QWidget()
        container.setLayout(self.mouse_position_layout)
        self.setCentralWidget(container)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create status label and add to status bar
        self.auto_label_mode = False
        self.status_label = QLabel("Manual labeling")
        self.status_bar.addWidget(self.status_label)
        
        self.image_files = []
        self.current_grayscale_image = []
        self.current_binary_image = []
        self.current_image_index = 0
        self.current_threshold = 128  # Default threshold value
        self.setFocusPolicy(Qt.StrongFocus)
        self.class_labels = {}
        self.image_folder = None

        # Connect the remove_all_rectangles_action after initializing self.canvas
        self.remove_all_rectangles_action.triggered.connect(self.canvas.remove_all_rectangles)

    def load_class_labels(self):
        file_path = QFileDialog.getOpenFileName(self, "Open Class Labels File", "", "Text Files (*.txt)")[0]
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.class_labels = {}
                for line in file:
                    key, class_label = line.strip().split(',')
                    self.class_labels[key.lower()] = class_label
            self.canvas.class_labels = self.class_labels
            self.open_folder_action.setEnabled(True)  # Enable folder selection
            self.save_xml_action.setEnabled(True)  # Enable save XML
            self.load_xml_action.setEnabled(True)  # Enable load XML

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Folder", "")
        if folder_path:
            self.image_folder = folder_path
            self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            self.image_files = [os.path.normpath(f).replace("\\", "/") for f in self.image_files]  # Normalize paths
            if self.image_files:
                self.current_image_index = 0
                self.load_image()
                self.generate_yolo_action.setEnabled(True)  # Enable generate YOLO files

    def update_status_label(self):
        mode = f"Auto labeling  (threshold = {self.current_threshold})" if self.auto_label_mode else "Manual labeling                                "
        self.status_label.setText(mode)

    def load_image(self):
        if self.image_files:
            image_path = self.image_files[self.current_image_index]
            self.current_grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            pixmap = QPixmap(image_path)
            self.canvas.setPixmap(pixmap)
            threshold = self.canvas.load_labels_from_xml(image_path)
            if threshold is not None:
                self.current_threshold = threshold
            if self.auto_label_mode:
                self.binarize_image()
            else:
                self.grayscale_image()
            self.update_status_label()
            self.update_remove_all_rectangles_action()
            self.setWindowTitle(f"Image Labeling Tool  -  [{self.current_image_index + 1} / {len(self.image_files)}]  -   {os.path.basename(image_path)}")

    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.canvas.save_labels(self.image_files[self.current_image_index], self.current_threshold)
            self.current_image_index += 1
            self.load_image()

    def previous_image(self):
        if self.image_files and self.current_image_index > 0:
            self.canvas.save_labels(self.image_files[self.current_image_index], self.current_threshold)
            self.current_image_index -= 1
            self.load_image()

    def save_xml(self):
        if self.image_files:
            self.canvas.save_labels(self.image_files[self.current_image_index], self.current_threshold)

    def load_xml(self):
        if self.image_files:
            self.canvas.load_labels_from_xml(self.image_files[self.current_image_index])

    def generate_yolo_files(self):
        if self.image_folder:
            xml_files = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.lower().endswith('.xml')]
            for xml_file in xml_files:
                self.generate_yolo_from_xml(xml_file)

    def generate_yolo_from_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        yolo_label_path = os.path.splitext(xml_file)[0] + ".txt"
        yolo_lines = []

        for obj in root.findall("object"):
            class_char = obj.find("name").text
            class_index = list(self.class_labels.values()).index(class_char)
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            bbox_width = (xmax - xmin)
            bbox_height = (ymax - ymin)

            yolo_line = f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}"
            yolo_lines.append(yolo_line)

        with open(yolo_label_path, 'w') as yolo_file:
            yolo_file.write("\n".join(yolo_lines))

    def increase_threshold(self):
        if self.auto_label_mode:
            self.current_threshold = min(self.current_threshold + 1, 255)
            self.update_status_label()
            self.binarize_image()

    def decrease_threshold(self):
        if self.auto_label_mode:
            self.current_threshold = max(self.current_threshold - 1, 0)
            self.update_status_label()
            self.binarize_image()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.next_image()

        elif event.key() == Qt.Key_Left:
            self.previous_image()

        elif event.key() == Qt.Key_Slash:
            self.auto_label_mode = not self.auto_label_mode
            self.update_status_label()
            if self.auto_label_mode:
                self.binarize_image()
            else:
                self.grayscale_image()

        elif event.key() == Qt.Key_Plus:
            self.increase_threshold()

        elif event.key() == Qt.Key_Minus:
            self.decrease_threshold()

        elif event.key() == Qt.Key_Delete:
            self.canvas.check_mouse_in_rectangle(self.canvas.start_point)
            self.remove_rectangle()

        else:
            if self.auto_label_mode:
                self.canvas.detect_white_area(self.canvas.start_point)
                self.canvas.drawing = False
                self.canvas.check_mouse_in_rectangle(self.canvas.start_point)
                self.canvas.keyPressEvent(event)
            else:
                self.canvas.check_mouse_in_rectangle(self.canvas.start_point)
                self.canvas.keyPressEvent(event)


    def remove_rectangle(self):
        if self.canvas.current_rect:
            for i, (rect, _, _) in enumerate(self.canvas.rectangles):
                if rect == self.canvas.current_rect:
                    del self.canvas.rectangles[i]
                    self.canvas.update()
                    self.update_remove_all_rectangles_action()

    def update_remove_all_rectangles_action(self):
        self.remove_all_rectangles_action.setEnabled(bool(self.canvas.rectangles))

    def grayscale_image(self):
        height, width = self.current_grayscale_image.shape
        q_image = QImage(self.current_grayscale_image.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.canvas.setPixmap(pixmap.scaled(self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def binarize_image(self):
        image_path = self.image_files[self.current_image_index]
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.current_grayscale_image = self.image
        if self.image is None:
            print(f"Error loading image: {image_path}")
            return
        _, self.binary_image = cv2.threshold(self.image, self.current_threshold, 255, cv2.THRESH_BINARY)
        if self.binary_image is None:
            print(f"Error in thresholding image: {image_path}")
            return
        height, width = self.binary_image.shape
        q_image = QImage(self.binary_image.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.canvas.setPixmap(pixmap.scaled(self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == "__main__":
    app = QApplication([])
    window = ImageLabelingTool()
    window.show()
    app.exec_()