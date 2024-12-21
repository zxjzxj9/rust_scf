#! /usr/bin/env python3

import sys
import requests
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QLineEdit, QListWidget, QPushButton, QLabel, 
                              QMessageBox, QProgressDialog, QDialog, QTextEdit,
                              QDialogButtonBox, QHBoxLayout, QTabWidget)
from PySide6.QtCore import Qt, QThread, Signal

class DataFetcher(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def run(self):
        try:
            base_url = "https://www.basissetexchange.org"
            response = requests.get(base_url + '/api/metadata')
            self.finished.emit(response.json())
        except Exception as e:
            self.error.emit(str(e))

class MetadataDialog(QDialog):
    def __init__(self, basis_name, basis_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Basis Set Information: {basis_name}")
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout(self)
        
        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # Overview Tab
        overview_widget = QWidget()
        overview_layout = QVBoxLayout(overview_widget)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        
        # Format basic information
        info_html = f"<h2>{basis_name}</h2>"
        if isinstance(basis_info, dict):
            for key, value in basis_info.items():
                if key != 'elements' and value:  # Skip elements list and empty values
                    info_html += f"<p><b>{key.title()}:</b> {value}</p>"
        
        info_text.setHtml(info_html)
        overview_layout.addWidget(info_text)
        tab_widget.addTab(overview_widget, "Overview")

        # Elements Tab
        if isinstance(basis_info, dict) and 'elements' in basis_info:
            elements_widget = QWidget()
            elements_layout = QVBoxLayout(elements_widget)
            
            elements_text = QTextEdit()
            elements_text.setReadOnly(True)
            
            # Format elements information
            elements_html = "<h3>Supported Elements</h3>"
            if isinstance(basis_info['elements'], dict):
                for element, data in basis_info['elements'].items():
                    elements_html += f"<p><b>{element}:</b> {data}</p>"
            elif isinstance(basis_info['elements'], list):
                elements_html += "<p>" + ", ".join(map(str, basis_info['elements'])) + "</p>"
            
            elements_text.setHtml(elements_html)
            elements_layout.addWidget(elements_text)
            tab_widget.addTab(elements_widget, "Elements")

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        button_gen = QPushButton("Generate Basis Set File")
        button_gen.clicked.connect(self.generate_basis_file)
        layout.addWidget(button_gen)

    def generate_basis_file(self):
        QMessageBox.information(self, "Basis Set File", "File generated successfully")

class BasisSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basis Set Selector")
        self.setMinimumSize(600, 400)
        
        self.basis_data = {}
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Search area
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Type to filter basis sets...")
        self.search_box.textChanged.connect(self.filter_items)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box)
        layout.addLayout(search_layout)
        
        # List widget
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.handle_selection)
        layout.addWidget(self.list_widget)
        
        # Status and buttons
        self.status_label = QLabel("Loading basis sets...")
        layout.addWidget(self.status_label)
        
        button_layout = QHBoxLayout()
        self.select_button = QPushButton("Select")
        self.select_button.clicked.connect(self.handle_selection)
        button_layout.addWidget(self.select_button)
        layout.addLayout(button_layout)
        
        self.fetch_data()

    def fetch_data(self):
        self.progress = QProgressDialog("Fetching basis sets...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()
        
        self.fetcher = DataFetcher()
        self.fetcher.finished.connect(self.handle_data)
        self.fetcher.error.connect(self.handle_error)
        self.fetcher.start()

    def handle_data(self, data):
        self.progress.close()
        self.basis_data = data
        self.populate_list()
        self.status_label.setText(f"Total basis sets: {len(data)}")

    def handle_error(self, error_msg):
        self.progress.close()
        QMessageBox.critical(self, "Error", f"Failed to fetch basis sets: {error_msg}")
        self.status_label.setText("Error loading basis sets")

    def populate_list(self, filter_text=""):
        self.list_widget.clear()
        filter_text = filter_text.lower()
        
        for basis_name in sorted(self.basis_data.keys()):
            if filter_text in basis_name.lower():
                self.list_widget.addItem(basis_name)

    def filter_items(self, text):
        self.populate_list(text)
        matching_count = self.list_widget.count()
        total_count = len(self.basis_data)
        self.status_label.setText(f"Showing {matching_count} of {total_count} basis sets")

    def handle_selection(self):
        current_item = self.list_widget.currentItem()
        if current_item:
            basis_name = current_item.text()
            basis_info = self.basis_data[basis_name]
            
            dialog = MetadataDialog(basis_name, basis_info, self)
            if dialog.exec() == QDialog.Accepted:
                print(f"Selected basis set: {basis_name}")
                self.close()

def main():
    app = QApplication(sys.argv)
    window = BasisSelector()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
