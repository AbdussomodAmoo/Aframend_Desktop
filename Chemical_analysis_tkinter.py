import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint
from typing import List, Dict, Optional, Union
import joblib
import warnings
import os
import requests
from datetime import datetime
from groq import Groq
import threading
from PIL import Image, ImageTk
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

warnings.filterwarnings('ignore')

class ChemicalAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chemical Analysis Platform - Desktop")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.groq_client = None
        self.models = {}
        self.bioactivity_model = None
        self.bioactivity_scaler = None
        self.results_df = None
        self.selected_endpoints = []
        
        # Constants from original code
        self.TOX21_ENDPOINTS = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
        self.ENDPOINT_NAMES = {
            'NR-AR': 'Androgen Receptor Disruption',
            'NR-AR-LBD': 'Androgen Receptor Binding',
            'NR-AhR': 'Aryl Hydrocarbon Receptor',
            'NR-Aromatase': 'Aromatase Inhibition',
            'NR-ER': 'Estrogen Receptor Disruption',
            'NR-ER-LBD': 'Estrogen Receptor Binding',
            'NR-PPAR-gamma': 'PPAR-gamma Activation',
            'SR-ARE': 'Antioxidant Response',
            'SR-ATAD5': 'DNA Damage Response',
            'SR-HSE': 'Heat Shock Response',
            'SR-MMP': 'Mitochondrial Toxicity',
            'SR-p53': 'p53 Tumor Suppressor'
        }
        
        self.setup_ui()
        self.load_models()
    
    def setup_ui(self):
        """Setup the main UI"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_toxicity_tab()
        self.create_bioactivity_tab()
        self.create_settings_tab()
        
    def create_toxicity_tab(self):
        """Create toxicity prediction tab"""
        self.toxicity_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.toxicity_frame, text="Toxicity Prediction")
        
        # Title
        title_label = tk.Label(self.toxicity_frame, text="Multi-Endpoint Toxicity Predictor", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Input frame
        input_frame = ttk.LabelFrame(self.toxicity_frame, text="SMILES Input", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)
        
        # Input method selection
        self.tox_input_method = tk.StringVar(value="Single SMILES")
        input_methods = ["Single SMILES", "Multiple SMILES", "CSV File"]
        
        for i, method in enumerate(input_methods):
            rb = ttk.Radiobutton(input_frame, text=method, variable=self.tox_input_method, 
                               value=method, command=self.update_tox_input_method)
            rb.grid(row=0, column=i, padx=10, pady=5)
        
        # Input widgets frame
        self.tox_input_widgets_frame = ttk.Frame(input_frame)
        self.tox_input_widgets_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)
        
        self.update_tox_input_method()
        
        # Endpoint selection frame
        endpoint_frame = ttk.LabelFrame(self.toxicity_frame, text="Select Endpoints", padding=10)
        endpoint_frame.pack(fill="x", padx=10, pady=5)
        
        # Select all checkbox
        self.tox_select_all = tk.BooleanVar(value=True)
        select_all_cb = ttk.Checkbutton(endpoint_frame, text="Select All", 
                                       variable=self.tox_select_all,
                                       command=self.toggle_all_endpoints)
        select_all_cb.pack(anchor="w")
        
        # Endpoint checkboxes frame
        self.endpoint_frame = ttk.Frame(endpoint_frame)
        self.endpoint_frame.pack(fill="x", pady=5)
        
        self.endpoint_vars = {}
        self.create_endpoint_checkboxes()
        
        # Predict button
        predict_button = ttk.Button(self.toxicity_frame, text="Predict Toxicity", 
                                   command=self.predict_toxicity, style="Accent.TButton")
        predict_button.pack(pady=10)
        
        # Results frame
        self.tox_results_frame = ttk.LabelFrame(self.toxicity_frame, text="Results", padding=10)
        self.tox_results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Results will be populated dynamically
        
    def create_bioactivity_tab(self):
        """Create bioactivity prediction tab"""
        self.bioactivity_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.bioactivity_frame, text="IC50 Bioactivity")
        
        # Title
        title_label = tk.Label(self.bioactivity_frame, text="IC50 Bioactivity Predictor", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Model loading frame
        model_frame = ttk.LabelFrame(self.bioactivity_frame, text="Model Loading", padding=10)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        # Model source selection
        self.bio_model_source = tk.StringVar(value="Local File")
        ttk.Radiobutton(model_frame, text="Local File", variable=self.bio_model_source, 
                       value="Local File", command=self.update_bio_model_method).grid(row=0, column=0, padx=10)
        ttk.Radiobutton(model_frame, text="GitHub URL", variable=self.bio_model_source, 
                       value="GitHub URL", command=self.update_bio_model_method).grid(row=0, column=1, padx=10)
        
        # Model loading widgets frame
        self.bio_model_widgets_frame = ttk.Frame(model_frame)
        self.bio_model_widgets_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.update_bio_model_method()
        
        # Input frame
        input_frame = ttk.LabelFrame(self.bioactivity_frame, text="SMILES Input", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)
        
        # Input method selection
        self.bio_input_method = tk.StringVar(value="Single SMILES")
        input_methods = ["Single SMILES", "Multiple SMILES", "CSV File"]
        
        for i, method in enumerate(input_methods):
            rb = ttk.Radiobutton(input_frame, text=method, variable=self.bio_input_method, 
                               value=method, command=self.update_bio_input_method)
            rb.grid(row=0, column=i, padx=10, pady=5)
        
        # Input widgets frame
        self.bio_input_widgets_frame = ttk.Frame(input_frame)
        self.bio_input_widgets_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)
        
        self.update_bio_input_method()
        
        # Predict button
        predict_button = ttk.Button(self.bioactivity_frame, text="Predict IC50", 
                                   command=self.predict_bioactivity, style="Accent.TButton")
        predict_button.pack(pady=10)
        
        # Results frame
        self.bio_results_frame = ttk.LabelFrame(self.bioactivity_frame, text="Results", padding=10)
        self.bio_results_frame.pack(fill="both", expand=True, padx=10, pady=5)
    
    def create_settings_tab(self):
        """Create settings tab"""
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings & AI")
        
        # AI Settings frame
        ai_frame = ttk.LabelFrame(self.settings_frame, text="AI Analysis Settings", padding=10)
        ai_frame.pack(fill="x", padx=10, pady=10)
        
        # Groq API Key
        tk.Label(ai_frame, text="Groq API Key:").grid(row=0, column=0, sticky="w", pady=5)
        self.groq_api_key_var = tk.StringVar()
        api_key_entry = ttk.Entry(ai_frame, textvariable=self.groq_api_key_var, width=50, show="*")
        api_key_entry.grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Button(ai_frame, text="Set API Key", command=self.set_groq_api_key).grid(row=0, column=2, padx=10)
        
        # Model Status frame
        status_frame = ttk.LabelFrame(self.settings_frame, text="Model Status", padding=10)
        status_frame.pack(fill="x", padx=10, pady=10)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=10, width=80)
        self.status_text.pack(fill="both", expand=True)
        
        # About frame
        about_frame = ttk.LabelFrame(self.settings_frame, text="About", padding=10)
        about_frame.pack(fill="x", padx=10, pady=10)
        
        about_text = """Chemical Analysis Platform - Desktop Version
        
Features:
- Multi-endpoint toxicity prediction using TOX21 models
- IC50 bioactivity prediction for drug discovery
- AI-powered analysis with Groq API integration
- Support for single compounds, multiple SMILES, and CSV files
- Optimized for Snapdragon X Elite processors
        
Compatible with Windows ARM64 and other ARM-based systems."""
        
        tk.Label(about_frame, text=about_text, justify="left").pack(anchor="w")
    
    def create_endpoint_checkboxes(self):
        """Create checkboxes for endpoint selection"""
        # Clear existing checkboxes
        for widget in self.endpoint_frame.winfo_children():
            widget.destroy()
        
        # Create checkboxes in a grid
        row = 0
        col = 0
        max_cols = 3
        
        for endpoint in self.TOX21_ENDPOINTS:
            if endpoint not in self.endpoint_vars:
                self.endpoint_vars[endpoint] = tk.BooleanVar(value=True)
            
            cb = ttk.Checkboxbutton(self.endpoint_frame, 
                                   text=f"{endpoint}: {self.ENDPOINT_NAMES[endpoint][:30]}...",
                                   variable=self.endpoint_vars[endpoint])
            cb.grid(row=row, column=col, sticky="w", padx=5, pady=2)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    
    def toggle_all_endpoints(self):
        """Toggle all endpoint selections"""
        select_all = self.tox_select_all.get()
        for var in self.endpoint_vars.values():
            var.set(select_all)
    
    def update_tox_input_method(self):
        """Update toxicity input method widgets"""
        # Clear existing widgets
        for widget in self.tox_input_widgets_frame.winfo_children():
            widget.destroy()
        
        method = self.tox_input_method.get()
        
        if method == "Single SMILES":
            tk.Label(self.tox_input_widgets_frame, text="Enter SMILES:").grid(row=0, column=0, sticky="w")
            self.tox_single_smiles = tk.StringVar()
            entry = ttk.Entry(self.tox_input_widgets_frame, textvariable=self.tox_single_smiles, width=60)
            entry.grid(row=0, column=1, padx=10)
            
        elif method == "Multiple SMILES":
            tk.Label(self.tox_input_widgets_frame, text="Enter SMILES (one per line):").grid(row=0, column=0, sticky="nw")
            self.tox_multi_smiles = scrolledtext.ScrolledText(self.tox_input_widgets_frame, width=60, height=8)
            self.tox_multi_smiles.grid(row=0, column=1, padx=10)
            
        elif method == "CSV File":
            tk.Label(self.tox_input_widgets_frame, text="CSV File:").grid(row=0, column=0, sticky="w")
            self.tox_csv_path = tk.StringVar()
            ttk.Entry(self.tox_input_widgets_frame, textvariable=self.tox_csv_path, width=50).grid(row=0, column=1, padx=10)
            ttk.Button(self.tox_input_widgets_frame, text="Browse", 
                      command=self.browse_tox_csv).grid(row=0, column=2, padx=5)
    
    def update_bio_input_method(self):
        """Update bioactivity input method widgets"""
        # Clear existing widgets
        for widget in self.bio_input_widgets_frame.winfo_children():
            widget.destroy()
        
        method = self.bio_input_method.get()
        
        if method == "Single SMILES":
            tk.Label(self.bio_input_widgets_frame, text="Enter SMILES:").grid(row=0, column=0, sticky="w")
            self.bio_single_smiles = tk.StringVar()
            entry = ttk.Entry(self.bio_input_widgets_frame, textvariable=self.bio_single_smiles, width=60)
            entry.grid(row=0, column=1, padx=10)
            
        elif method == "Multiple SMILES":
            tk.Label(self.bio_input_widgets_frame, text="Enter SMILES (one per line):").grid(row=0, column=0, sticky="nw")
            self.bio_multi_smiles = scrolledtext.ScrolledText(self.bio_input_widgets_frame, width=60, height=8)
            self.bio_multi_smiles.grid(row=0, column=1, padx=10)
            
        elif method == "CSV File":
            tk.Label(self.bio_input_widgets_frame, text="CSV File:").grid(row=0, column=0, sticky="w")
            self.bio_csv_path = tk.StringVar()
            ttk.Entry(self.bio_input_widgets_frame, textvariable=self.bio_csv_path, width=50).grid(row=0, column=1, padx=10)
            ttk.Button(self.bio_input_widgets_frame, text="Browse", 
                      command=self.browse_bio_csv).grid(row=0, column=2, padx=5)
    
    def update_bio_model_method(self):
        """Update bioactivity model loading method widgets"""
        # Clear existing widgets
        for widget in self.bio_model_widgets_frame.winfo_children():
            widget.destroy()
        
        method = self.bio_model_source.get()
        
        if method == "Local File":
            ttk.Button(self.bio_model_widgets_frame, text="Load Local Model", 
                      command=self.load_local_bioactivity_model).grid(row=0, column=0, padx=10)
            
        elif method == "GitHub URL":
            tk.Label(self.bio_model_widgets_frame, text="Model URL:").grid(row=0, column=0, sticky="w")
            self.bio_model_url = tk.StringVar()
            ttk.Entry(self.bio_model_widgets_frame, textvariable=self.bio_model_url, width=50).grid(row=0, column=1, padx=10)
            
            tk.Label(self.bio_model_widgets_frame, text="Scaler URL (optional):").grid(row=1, column=0, sticky="w")
            self.bio_scaler_url = tk.StringVar()
            ttk.Entry(self.bio_model_widgets_frame, textvariable=self.bio_scaler_url, width=50).grid(row=1, column=1, padx=10)
            
            ttk.Button(self.bio_model_widgets_frame, text="Load from GitHub", 
                      command=self.load_github_bioactivity_model).grid(row=2, column=0, columnspan=2, pady=10)
    
    def browse_tox_csv(self):
        """Browse for toxicity CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.tox_csv_path.set(filename)
    
    def browse_bio_csv(self):
        """Browse for bioactivity CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.bio_csv_path.set(filename)
    
    def set_groq_api_key(self):
        """Set Groq API key"""
        api_key = self.groq_api_key_var.get().strip()
        if api_key:
            try:
                self.groq_client = Groq(api_key=api_key)
                messagebox.showinfo("Success", "Groq API key set successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to set API key: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please enter a valid API key")
    
    def load_models(self):
        """Load toxicity models"""
        def load_in_thread():
            loaded_models = {}
            missing_models = []
            
            for endpoint in self.TOX21_ENDPOINTS:
                try:
                    loaded_models[endpoint] = joblib.load(f'{endpoint}.pkl')
                    self.update_status(f"Loaded {endpoint} model")
                except FileNotFoundError:
                    missing_models.append(endpoint)
                except Exception as e:
                    self.update_status(f"Error loading {endpoint} model: {str(e)}")
                    missing_models.append(endpoint)
            
            self.models = loaded_models
            
            if missing_models:
                self.update_status(f"Missing models: {', '.join(missing_models)}")
            
            if loaded_models:
                self.update_status(f"Successfully loaded {len(loaded_models)} toxicity models!")
        
        # Load models in a separate thread to avoid blocking UI
        threading.Thread(target=load_in_thread, daemon=True).start()
    
    def load_local_bioactivity_model(self):
        """Load local bioactivity model"""
        try:
            self.bioactivity_model = joblib.load('bioactivity_model.joblib')
            self.update_status("Bioactivity model loaded successfully from local file!")
            
            # Try to load scaler
            try:
                self.bioactivity_scaler = joblib.load('scaler_X.joblib')
                self.update_status("Bioactivity scaler loaded successfully!")
            except:
                self.bioactivity_scaler = self.create_default_scaler()
                self.update_status("No scaler found, using identity transformation")
                
        except FileNotFoundError:
            messagebox.showerror("Error", "Model file 'bioactivity_model.joblib' not found in current directory.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading local model: {str(e)}")
    
    def load_github_bioactivity_model(self):
        """Load bioactivity model from GitHub"""
        model_url = self.bio_model_url.get().strip()
        scaler_url = self.bio_scaler_url.get().strip() or None
        
        if not model_url:
            messagebox.showwarning("Warning", "Please enter a model URL")
            return
        
        def load_in_thread():
            try:
                self.update_status("Loading model from GitHub...")
                model, scaler = self.load_bioactivity_model_and_scaler_from_github(model_url, scaler_url)
                self.bioactivity_model = model
                self.bioactivity_scaler = scaler if scaler else self.create_default_scaler()
                self.update_status("Model loaded successfully from GitHub!")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=load_in_thread, daemon=True).start()
    
    def update_status(self, message):
        """Update status text (thread-safe)"""
        def update():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.status_text.see(tk.END)
        
        self.root.after(0, update)
    
    def get_tox_smiles_list(self):
        """Get SMILES list for toxicity prediction"""
        method = self.tox_input_method.get()
        
        if method == "Single SMILES":
            smiles = self.tox_single_smiles.get().strip()
            return [smiles] if smiles else []
            
        elif method == "Multiple SMILES":
            text = self.tox_multi_smiles.get("1.0", tk.END).strip()
            return [s.strip() for s in text.split('\n') if s.strip()]
            
        elif method == "CSV File":
            csv_path = self.tox_csv_path.get().strip()
            if not csv_path:
                return []
            
            try:
                df = pd.read_csv(csv_path)
                # Find SMILES column
                smiles_col = None
                for col in df.columns:
                    if col.lower() in ['canonical_smiles', 'smiles', 'smile']:
                        smiles_col = col
                        break
                
                if smiles_col:
                    return df[smiles_col].dropna().tolist()
                else:
                    messagebox.showerror("Error", "No SMILES column found in CSV file")
                    return []
            except Exception as e:
                messagebox.showerror("Error", f"Error reading CSV file: {str(e)}")
                return []
        
        return []
    
    def get_bio_smiles_list(self):
        """Get SMILES list for bioactivity prediction"""
        method = self.bio_input_method.get()
        
        if method == "Single SMILES":
            smiles = self.bio_single_smiles.get().strip()
            return [smiles] if smiles else []
            
        elif method == "Multiple SMILES":
            text = self.bio_multi_smiles.get("1.0", tk.END).strip()
            return [s.strip() for s in text.split('\n') if s.strip()]
            
        elif method == "CSV File":
            csv_path = self.bio_csv_path.get().strip()
            if not csv_path:
                return []
            
            try:
                df = pd.read_csv(csv_path)
                # Find SMILES column
                smiles_col = None
                for col in df.columns:
                    if col.lower() in ['canonical_smiles', 'smiles', 'smile']:
                        smiles_col = col
                        break
                
                if smiles_col:
                    return df[smiles_col].dropna().tolist()
                else:
                    messagebox.showerror("Error", "No SMILES column found in CSV file")
                    return []
            except Exception as e:
                messagebox.showerror("Error", f"Error reading CSV file: {str(e)}")
                return []
        
        return []
    
    def get_selected_endpoints(self):
        """Get selected endpoints"""
        selected = []
        for endpoint, var in self.endpoint_vars.items():
            if var.get():
                selected.append(endpoint)
        return selected
    
    def predict_toxicity(self):
        """Predict toxicity"""
        smiles_list = self.get_tox_smiles_list()
        if not smiles_list:
            messagebox.showwarning("Warning", "Please enter SMILES data")
            return
        
        selected_endpoints = self.get_selected_endpoints()
        if not selected_endpoints:
            messagebox.showwarning("Warning", "Please select at least one endpoint")
            return
        
        if not self.models:
            messagebox.showerror("Error", "No models loaded")
            return
        
        def predict_in_thread():
            try:
                self.update_status("Making toxicity predictions...")
                results_df = self.predict_toxicity_multi_endpoint(smiles_list, self.models, selected_endpoints)
                self.display_toxicity_results(results_df, selected_endpoints)
                self.update_status("Toxicity predictions completed!")
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {str(e)}")
        
        threading.Thread(target=predict_in_thread, daemon=True).start()
    
    def predict_bioactivity(self):
        """Predict bioactivity"""
        if not self.bioactivity_model:
            messagebox.showerror("Error", "Please load a bioactivity model first")
            return
        
        smiles_list = self.get_bio_smiles_list()
        if not smiles_list:
            messagebox.showwarning("Warning", "Please enter SMILES data")
            return
        
        def predict_in_thread():
            try:
                self.update_status("Making IC50 predictions...")
                results_df = self.predict_ic50(smiles_list, self.bioactivity_model, self.bioactivity_scaler)
                self.display_bioactivity_results(results_df)
                self.update_status("IC50 predictions completed!")
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {str(e)}")
        
        threading.Thread(target=predict_in_thread, daemon=True).start()
    
    def display_toxicity_results(self, results_df, selected_endpoints):
        """Display toxicity results"""
        # Clear existing results
        for widget in self.tox_results_frame.winfo_children():
            widget.destroy()
        
        # Create results display
        # Summary metrics
        summary_frame = ttk.Frame(self.tox_results_frame)
        summary_frame.pack(fill="x", pady=5)
        
        valid_results = results_df[results_df['Valid'] == True]
        
        tk.Label(summary_frame, text=f"Total Compounds: {len(results_df)}", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        tk.Label(summary_frame, text=f"Valid Predictions: {len(valid_results)}", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        tk.Label(summary_frame, text=f"Endpoints Tested: {len(selected_endpoints)}", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        
        # Results table
        table_frame = ttk.Frame(self.tox_results_frame)
        table_frame.pack(fill="both", expand=True, pady=10)
        
        # Create treeview for results
        columns = ['SMILES'] + [f'{ep}_Prediction' for ep in selected_endpoints[:5]]  # Limit columns for display
        
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Populate table
        for idx, row in results_df.iterrows():
            values = [row['SMILES'][:30] + '...' if len(row['SMILES']) > 30 else row['SMILES']]
            for ep in selected_endpoints[:5]:  # Limit to first 5 endpoints for display
                pred_col = f'{ep}_Prediction'
                if pred_col in row:
                    values.append(str(row[pred_col]))
                else:
                    values.append('N/A')
            tree.insert('', 'end', values=values)
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.tox_results_frame)
        buttons_frame.pack(fill="x", pady=10)
        
        ttk.Button(buttons_frame, text="Export Results", 
                  command=lambda: self.export_results(results_df, "toxicity")).pack(side="left", padx=10)
        
        if self.groq_client:
            ttk.Button(buttons_frame, text="Generate AI Analysis", 
                      command=lambda: self.generate_toxicity_analysis(results_df, selected_endpoints)).pack(side="left", padx=10)
    
    def display_bioactivity_results(self, results_df):
        """Display bioactivity results"""
        # Clear existing results
        for widget in self.bio_results_frame.winfo_children():
            widget.destroy()
        
        # Summary metrics
        summary_frame = ttk.Frame(self.bio_results_frame)
        summary_frame.pack(fill="x", pady=5)
        
        valid_results = results_df[results_df['Valid'] == True]
        
        tk.Label(summary_frame, text=f"Total Compounds: {len(results_df)}", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        tk.Label(summary_frame, text=f"Valid Predictions: {len(valid_results)}", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        
        if len(valid_results) > 0:
            highly_active = len(valid_results[valid_results['Activity_Level'] == 'Highly Active'])
            avg_ic50 = valid_results['Predicted_IC50_Scaled'].mean()
            tk.Label(summary_frame, text=f"Highly Active: {highly_active}", font=("Arial", 10, "bold")).pack(side="left", padx=10)
            tk.Label(summary_frame, text=f"Avg IC50: {avg_ic50:.3f}", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        
        # Results table
        table_frame = ttk.Frame(self.bio_results_frame)
        table_frame.pack(fill="both", expand=True, pady=10)
        
        # Create treeview for results
        columns = ['SMILES', 'IC50_Scaled', 'Activity_Level', 'Properties']
        
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Populate table
        for idx, row in results_df.iterrows():
            values = [
                row['SMILES'][:30] + '...' if len(row['SMILES']) > 30 else row['SMILES'],
                str(row['Predicted_IC50_Scaled']) if row['Valid'] else 'N/A',
                str(row['Activity_Level']),
                str(row['Molecular_Properties'])
            ]
            tree.insert('', 'end', values=values)
        
        # Visualization frame
        if len(valid_results) > 0:
            viz_frame = ttk.LabelFrame(self.bio_results_frame, text="Visualization", padding=10)
            viz_frame.pack(fill="x", pady=10)
            
            # Create matplotlib figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Activity level distribution
            activity_counts = valid_results['Activity_Level'].value_counts()
            ax1.bar(activity_counts.index, activity_counts.values)
            ax1.set_title('Activity Level Distribution')
            ax1.set_xlabel('Activity Level')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            # IC50 distribution
            ax2.hist(valid_results['Predicted_IC50_Scaled'], bins=20, alpha=0.7)
            ax2.set_title('IC50 Distribution')
            ax2.set_xlabel('Scaled IC50')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.bio_results_frame)
        buttons_frame.pack(fill="x", pady=10)
        
        ttk.Button(buttons_frame, text="Export Results", 
                  command=lambda: self.export_results(results_df, "bioactivity")).pack(side="left", padx=10)
        
        if self.groq_client:
            ttk.Button(buttons_frame, text="Generate AI Analysis", 
                      command=lambda: self.generate_bioactivity_analysis(results_df)).pack(side="left", padx=10)
    
    def export_results(self, results_df, analysis_type):
        """Export results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{analysis_type}_predictions_{timestamp}.csv"
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialvalue=default_filename
        )
        
        if filename:
            try:
                results_df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def generate_toxicity_analysis(self, results_df, selected_endpoints):
        """Generate AI analysis for toxicity results"""
        if not self.groq_client:
            messagebox.showwarning("Warning", "Please set your Groq API key first")
            return
        
        def generate_in_thread():
            try:
                self.update_status("Generating AI analysis...")
                
                # Prepare analysis data
                valid_results = results_df[results_df['Valid'] == True]
                total_compounds = len(results_df)
                valid_predictions = len(valid_results)
                endpoints_tested = len(selected_endpoints)
                
                # Create endpoint summaries
                endpoint_summaries = []
                for endpoint in selected_endpoints:
                    prob_col = f'{endpoint}_Probability'
                    if prob_col in results_df.columns:
                        valid_probs = results_df[prob_col].dropna()
                        if len(valid_probs) > 0:
                            toxic_count = len(valid_probs[valid_probs > 0.5])
                            endpoint_summaries.append(f"{endpoint}: {toxic_count}/{len(valid_probs)} toxic ({self.ENDPOINT_NAMES[endpoint]})")
                
                # Generate analysis
                system_prompt = """You are Dr. Sarah Chen, a senior toxicologist with 15+ years of experience in computational toxicology and drug safety assessment. Provide clear, accessible explanations of toxicity findings."""
                
                user_prompt = f"""
                ANALYSIS REQUEST: Multi-Endpoint Toxicity Assessment

                DATASET SUMMARY:
                - Total Compounds Analyzed: {total_compounds}
                - Valid Predictions: {valid_predictions}
                - Endpoints Tested: {endpoints_tested}

                ENDPOINT BREAKDOWN:
                {chr(10).join(endpoint_summaries)}

                Please provide:
                1. Overall risk assessment across all endpoints
                2. Most concerning endpoints and compounds
                3. Priority recommendations for risk management
                4. Limitations and next steps
                """
                
                analysis = self.generate_groq_response(system_prompt, user_prompt)
                self.show_analysis_window(analysis, "Toxicity Analysis")
                self.update_status("AI analysis completed!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate analysis: {str(e)}")
        
        threading.Thread(target=generate_in_thread, daemon=True).start()
    
    def generate_bioactivity_analysis(self, results_df):
        """Generate AI analysis for bioactivity results"""
        if not self.groq_client:
            messagebox.showwarning("Warning", "Please set your Groq API key first")
            return
        
        def generate_in_thread():
            try:
                self.update_status("Generating AI analysis...")
                
                valid_results = results_df[results_df['Valid'] == True]
                total_compounds = len(results_df)
                valid_predictions = len(valid_results)
                
                if len(valid_results) > 0:
                    highly_active = len(valid_results[valid_results['Activity_Level'] == 'Highly Active'])
                    active = len(valid_results[valid_results['Activity_Level'] == 'Active'])
                    avg_ic50 = valid_results['Predicted_IC50_Scaled'].mean()
                    
                    # Get top compounds
                    top_compounds = valid_results.nsmallest(5, 'Predicted_IC50_Scaled')
                    compounds_data = []
                    for idx, row in top_compounds.iterrows():
                        compounds_data.append(f"- {row['SMILES']}: IC50 {row['Predicted_IC50_Scaled']:.3f} ({row['Activity_Level']})")
                
                system_prompt = """You are Dr. Alex Chen, a computational medicinal chemist with 15+ years of experience in drug discovery and IC50 prediction. Explain IC50 predictions and their implications for drug development."""
                
                user_prompt = f"""
                ANALYSIS REQUEST: Compound Library IC50 Screening Analysis

                DATASET SUMMARY:
                - Total Compounds: {total_compounds}
                - Valid Predictions: {valid_predictions}
                - Highly Active: {highly_active}
                - Active: {active}
                - Average Scaled IC50: {avg_ic50:.3f}

                TOP COMPOUNDS:
                {chr(10).join(compounds_data)}

                Please provide:
                1. Overall library screening assessment
                2. Hit identification and prioritization
                3. Structure-activity patterns observed
                4. Lead optimization recommendations
                5. Next steps for medicinal chemistry teams
                """
                
                analysis = self.generate_groq_response(system_prompt, user_prompt)
                self.show_analysis_window(analysis, "IC50 Bioactivity Analysis")
                self.update_status("AI analysis completed!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate analysis: {str(e)}")
        
        threading.Thread(target=generate_in_thread, daemon=True).start()
    
    def generate_groq_response(self, system_prompt, user_prompt):
        """Generate response using Groq API"""
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama3-8b-8192",
                max_tokens=1500,
                temperature=0.7
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def show_analysis_window(self, analysis, title):
        """Show analysis in a new window"""
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title(title)
        analysis_window.geometry("800x600")
        
        # Create scrolled text widget
        text_widget = scrolledtext.ScrolledText(analysis_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Insert analysis text
        text_widget.insert("1.0", analysis)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Add save button
        save_button = ttk.Button(analysis_window, text="Save Analysis", 
                               command=lambda: self.save_analysis(analysis, title))
        save_button.pack(pady=10)
    
    def save_analysis(self, analysis, title):
        """Save analysis to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{title.replace(' ', '_').lower()}_{timestamp}.txt"
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialvalue=default_filename
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(analysis)
                messagebox.showinfo("Success", f"Analysis saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save analysis: {str(e)}")

    # ===== CORE PREDICTION FUNCTIONS =====
    
    def featurize(self, smiles):
        """Extract molecular features for bioactivity prediction"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        try:
            return {
                'MolWt': Descriptors.MolWt(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumRings': rdMolDescriptors.CalcNumRings(mol),
                'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
                'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
                'BertzCT': Descriptors.BertzCT(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol)
            }
        except Exception as e:
            return None
    
    def predict_ic50(self, smiles_list, model, scaler_X):
        """Make IC50 predictions"""
        results = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append({
                    'SMILES': smiles,
                    'Valid': False,
                    'Predicted_IC50_Scaled': None,
                    'Activity_Level': 'Invalid SMILES',
                    'Molecular_Properties': 'N/A'
                })
                continue
            
            features_dict = self.featurize(smiles)
            if features_dict is None:
                results.append({
                    'SMILES': smiles,
                    'Valid': False,
                    'Predicted_IC50_Scaled': None,
                    'Activity_Level': 'Feature extraction failed',
                    'Molecular_Properties': 'N/A'
                })
                continue
            
            try:
                feature_names = ['MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
                               'LogP', 'NumRings', 'HeavyAtomCount', 'FractionCSP3', 'NumAromaticRings',
                               'NumAliphaticRings', 'NumValenceElectrons', 'BertzCT', 'NumHeteroatoms']
                
                features_array = np.array([features_dict[name] for name in feature_names]).reshape(1, -1)
                features_scaled = scaler_X.transform(features_array)
                ic50_scaled = model.predict(features_scaled)[0]
                
                if ic50_scaled < -0.5:
                    activity_level = "Highly Active"
                elif ic50_scaled < 0:
                    activity_level = "Active"
                elif ic50_scaled < 0.5:
                    activity_level = "Moderately Active"
                else:
                    activity_level = "Low Activity"
                
                mol_props = f"MW: {features_dict['MolWt']:.1f}, LogP: {features_dict['LogP']:.2f}, TPSA: {features_dict['TPSA']:.1f}"
                
                results.append({
                    'SMILES': smiles,
                    'Valid': True,
                    'Predicted_IC50_Scaled': round(float(ic50_scaled), 3),
                    'Activity_Level': activity_level,
                    'Molecular_Properties': mol_props
                })
                
            except Exception as e:
                results.append({
                    'SMILES': smiles,
                    'Valid': False,
                    'Predicted_IC50_Scaled': None,
                    'Activity_Level': f'Prediction failed: {str(e)}',
                    'Molecular_Properties': 'N/A'
                })
        
        return pd.DataFrame(results)
    
    def extract_molecular_features(self, mol, include_fingerprints=True, morgan_radius=2, 
                                 morgan_bits=2048, include_fragments=True):
        """Extract comprehensive molecular features for toxicity prediction"""
        if mol is None:
            return {}

        features = {}

        try:
            # Basic molecular properties
            features['mol_weight'] = Descriptors.MolWt(mol)
            features['mol_logp'] = Descriptors.MolLogP(mol)
            features['tpsa'] = Descriptors.TPSA(mol)
            features['labute_asa'] = Descriptors.LabuteASA(mol)

            # Hydrogen bonding
            features['num_hbd'] = Descriptors.NumHDonors(mol)
            features['num_hba'] = Descriptors.NumHAcceptors(mol)
            features['max_partial_charge'] = Descriptors.MaxPartialCharge(mol)
            features['min_partial_charge'] = Descriptors.MinPartialCharge(mol)
            features['max_abs_partial_charge'] = Descriptors.MaxAbsPartialCharge(mol)

            # Structural features
            features['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            features['heavy_atom_count'] = Descriptors.HeavyAtomCount(mol)
            features['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            features['num_saturated_rings'] = Descriptors.NumSaturatedRings(mol)
            features['num_aliphatic_rings'] = Descriptors.NumAliphaticRings(mol)
            features['ring_count'] = Descriptors.RingCount(mol)
            features['fraction_csp3'] = Descriptors.FractionCSP3(mol)
            features['num_heteroatoms'] = Descriptors.NumHeteroatoms(mol)

            # Complexity measures
            features['bertz_ct'] = Descriptors.BertzCT(mol)
            features['hall_kier_alpha'] = Descriptors.HallKierAlpha(mol)
            features['kappa1'] = Descriptors.Kappa1(mol)
            features['kappa2'] = Descriptors.Kappa2(mol)
            features['kappa3'] = Descriptors.Kappa3(mol)

            # Drug-likeness
            features['qed'] = Descriptors.qed(mol)

            # VSA descriptors
            features['vsa_estate4'] = Descriptors.VSA_EState4(mol)
            features['vsa_estate9'] = Descriptors.VSA_EState9(mol)
            features['slogp_vsa4'] = Descriptors.SlogP_VSA4(mol)
            features['slogp_vsa6'] = Descriptors.SlogP_VSA6(mol)
            features['smr_vsa5'] = Descriptors.SMR_VSA5(mol)
            features['smr_vsa7'] = Descriptors.SMR_VSA7(mol)

            features['balaban_j'] = Descriptors.BalabanJ(mol)

            # Fragment counts
            if include_fragments:
                features['fr_phenol'] = Fragments.fr_phenol(mol)
                features['fr_benzene'] = Fragments.fr_benzene(mol)
                features['fr_halogen'] = Fragments.fr_halogen(mol)
                features['fr_ar_n'] = Fragments.fr_Ar_N(mol)
                features['fr_al_coo'] = Fragments.fr_Al_COO(mol)
                features['fr_alkyl_halide'] = Fragments.fr_alkyl_halide(mol)
                features['fr_amide'] = Fragments.fr_amide(mol)
                features['fr_aniline'] = Fragments.fr_aniline(mol)
                features['fr_nitro'] = Fragments.fr_nitro(mol)
                features['fr_sulfide'] = Fragments.fr_sulfide(mol)
                features['fr_ester'] = Fragments.fr_ester(mol)
                features['fr_ether'] = Fragments.fr_ether(mol)

            # Fingerprints
            if include_fingerprints:
                # Morgan fingerprints
                morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=morgan_radius, nBits=morgan_bits)
                for i, bit in enumerate(morgan_fp):
                    features[f'morgan_{i}'] = int(bit)

                # MACCS keys
                maccs_fp = GetMACCSKeysFingerprint(mol)
                for i, bit in enumerate(maccs_fp):
                    features[f'maccs_{i}'] = int(bit)

        except Exception as e:
            return {}

        return features
    
    def smiles_to_features(self, smiles):
        """Convert SMILES string to feature array for toxicity prediction"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        features_dict = self.extract_molecular_features(mol)
        if not features_dict:
            return None
        
        return np.array(list(features_dict.values()))
    
    def predict_toxicity_multi_endpoint(self, smiles_list, models, selected_endpoints):
        """Make toxicity predictions for multiple endpoints"""
        results = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            
            result_row = {
                'SMILES': smiles,
                'Valid': mol is not None
            }
            
            if mol is None:
                for endpoint in selected_endpoints:
                    result_row.update({
                        f'{endpoint}_Probability': None,
                        f'{endpoint}_Prediction': 'Invalid SMILES',
                        f'{endpoint}_Risk_Level': 'N/A'
                    })
                results.append(result_row)
                continue
            
            features = self.smiles_to_features(smiles)
            if features is None:
                for endpoint in selected_endpoints:
                    result_row.update({
                        f'{endpoint}_Probability': None,
                        f'{endpoint}_Prediction': 'Feature extraction failed',
                        f'{endpoint}_Risk_Level': 'N/A'
                    })
                results.append(result_row)
                continue
            
            features_reshaped = features.reshape(1, -1)
            
            for endpoint in selected_endpoints:
                if endpoint in models:
                    try:
                        pred_proba = models[endpoint].predict_proba(features_reshaped)[0]
                        
                        if len(pred_proba) == 1:
                            toxic_prob = pred_proba[0]
                        else:
                            toxic_prob = pred_proba[1]
                        
                        prediction = 'Toxic' if toxic_prob > 0.5 else 'Non-toxic'
                        risk_level = 'High' if toxic_prob > 0.7 else 'Medium' if toxic_prob > 0.3 else 'Low'
                        
                        result_row.update({
                            f'{endpoint}_Probability': round(toxic_prob, 3),
                            f'{endpoint}_Prediction': prediction,
                            f'{endpoint}_Risk_Level': risk_level
                        })
                        
                    except Exception as e:
                        result_row.update({
                            f'{endpoint}_Probability': None,
                            f'{endpoint}_Prediction': f'Prediction failed: {str(e)}',
                            f'{endpoint}_Risk_Level': 'N/A'
                        })
                else:
                    result_row.update({
                        f'{endpoint}_Probability': None,
                        f'{endpoint}_Prediction': 'Model not available',
                        f'{endpoint}_Risk_Level': 'N/A'
                    })
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def load_bioactivity_model_and_scaler_from_github(self, model_url, scaler_url=None):
        """Load bioactivity model and scaler from GitHub repository"""
        try:
            # Load model
            if 'raw.githubusercontent.com' in model_url:
                response = requests.get(model_url)
                if response.status_code == 200:
                    with open('temp_model.joblib', 'wb') as f:
                        f.write(response.content)
                    model = joblib.load('temp_model.joblib')
                    os.remove('temp_model.joblib')
                else:
                    raise Exception(f"Failed to download model: {response.status_code}")
            else:
                raise Exception("Please provide a direct raw GitHub URL to the model file")
            
            # Load scaler if URL provided
            scaler = None
            if scaler_url and 'raw.githubusercontent.com' in scaler_url:
                response = requests.get(scaler_url)
                if response.status_code == 200:
                    with open('temp_scaler.joblib', 'wb') as f:
                        f.write(response.content)
                    scaler = joblib.load('temp_scaler.joblib')
                    os.remove('temp_scaler.joblib')
            
            return model, scaler
        except Exception as e:
            raise Exception(f"Error loading from GitHub: {str(e)}")
    
    def create_default_scaler(self):
        """Create a default scaler when scaler is not available"""
        class IdentityScaler:
            def transform(self, X):
                return X
            def fit_transform(self, X):
                return X
        
        return IdentityScaler()

# Main application entry point
def main():
    # Set up matplotlib to work with tkinter
    plt.switch_backend('TkAgg')
    
    # Create root window
    root = tk.Tk()
    
    # Set application icon (optional)
    try:
        # You can add an icon file here if you have one
        # root.iconbitmap('app_icon.ico')
        pass
    except:
        pass
    
    # Create and run application
    app = ChemicalAnalysisApp(root)
    
    # Configure window closing behavior
    root.protocol("WM_DELETE_WINDOW", root.quit)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
