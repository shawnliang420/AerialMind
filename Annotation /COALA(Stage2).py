# AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import cv2
import numpy as np
import argparse
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from collections import defaultdict

class InteractiveMOTEditor:
    def __init__(self):
        # Initialize variables
        self.img_dir = None
        self.ann_file = None
        self.annotations = None
        self.image_files = None
        self.current_idx = 0
        self.current_frame_id = 0
        self.current_frame_bboxes = []
        self.show_labels = True  # Default show labels
        
        # Add auto-play related attributes
        self.is_auto_playing = False
        self.auto_play_speed = 200  # Milliseconds, control frame switching speed
        self.auto_play_job = None   # Store timer task ID
        
        # Mode selection
        self.editor_mode = None  # "merge" or "json"
        
        self.set_source_score_zero = False  # By default, the source object score is not modified
        
        # Track selections and their history
        self.persistent_selections = {}  # Dict of {object_id: first_frame_selected}
        self.selection_histories = {}    # Dict of {object_id: [frame_list]}
        
        # Merging state
        self.is_merging = False
        self.first_merge_box = None
        self.merged_boxes = []
        self.merged_source_ids = set()  # Track IDs of boxes that were used in merging
        self.auto_tracking_merge = False
        self.tracking_source_ids = []   # IDs being tracked for automatic merging [id1, id2]
        self.tracking_start_frame = 0   # Frame where tracking started
        self.tracking_merged_id = None  # ID of the merged box being tracked
        self.max_id = 0                 # Track maximum ID to ensure new IDs don't conflict
        
        # Image zoom variables
        self.zoom_scale = 1.0
        self.zoom_x = 0
        self.zoom_y = 0
        self.zoom_start_x = 0
        self.zoom_start_y = 0
        self.is_panning = False
        
        # Display mode
        self.show_zero_score = False  # Toggle for showing score=0 objects
        
        # Color scheme - soft and pleasant colors
        self.normal_color = (76, 175, 80)      # Soft green
        self.zero_score_color = (244, 67, 54)  # Bright but not harsh red
        self.selected_color = (255, 152, 0)    # Eye-catching orange
        self.merging_color = (255, 235, 59)    # Bright yellow
        self.merged_color = (156, 39, 176)     # Elegant purple
        
        # Interface theme colors
        self.theme_color = "#3F51B5"           # Indigo theme
        self.secondary_color = "#303F9F"       # Deep indigo secondary theme
        self.accent_color = "#FF4081"          # Pink accent color
        self.background_color = "#F5F5F5"      # Light gray background
        self.text_color = "#212121"            # Dark gray text
        self.light_text_color = "#757575"      # Light gray text
        
        # Font settings
        self.font_scale = 0.4
        self.font_thickness = 1
        self.title_font = ("Helvetica", 14, "bold")
        self.subtitle_font = ("Helvetica", 12, "bold")
        self.normal_font = ("Helvetica", 10)
        self.small_font = ("Helvetica", 9)
        
        # Output data format
        self.output_data = {
            "label": {},
            "ignore": {},
            "video_name": "",
            "sentence": ""
        }
        
        # Add fixed dataset directory and save directory
        self.dataset_default_dir = None  # Default directory for opening dataset
        self.save_default_dir = None     # Default directory for saving files
        
        # First prompt for mode selection, then initialize UI
        self.select_mode()
    
    def select_mode(self):
        """Display a dialog to select mode before initializing UI"""
        # Create a simple root window
        root = tk.Tk()
        root.title("Select Editor Mode")
        root.geometry("500x300")  # Increased height for credits
        
        # Center the window
        root.eval('tk::PlaceWindow . center')
        
        # Mode variable
        self.editor_mode = tk.StringVar(value="json")  # Default to JSON mode
        
        # Mode selection frame
        frame = ttk.Frame(root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Interactive RMOT Annotation Editor", font=("Helvetica", 14, "bold")).pack(pady=3)
        ttk.Label(frame, text="(Designed and developed by COALA)", font=("Helvetica", 8, "bold"),foreground="gray").pack(pady=3)
        
        ttk.Label(frame, text="Select Editor Mode:", font=("Helvetica", 12, "bold")).pack(pady=5)
        
        # Radio buttons for mode selection
        ttk.Radiobutton(frame, text="JSON Mode - Select objects for annotation", 
                        variable=self.editor_mode, value="json").pack(anchor=tk.W, pady=5)
        ttk.Label(frame, text="    (No merging, focus on creating annotation files)").pack(anchor=tk.W, padx=25)
        
        ttk.Radiobutton(frame, text="Merge Mode - Merge bounding boxes", 
                        variable=self.editor_mode, value="merge").pack(anchor=tk.W, pady=5)
        ttk.Label(frame, text="    (Automatic tracking of merged boxes)").pack(anchor=tk.W, padx=25)
        
        # Create a frame for the confirm button to ensure it's visible
        button_frame = ttk.Frame(frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        
        # Confirm button with larger size
        confirm_btn = ttk.Button(button_frame, text="Confirm", command=lambda: self.on_mode_selected(root))
        confirm_btn.pack(fill=tk.X, padx=50)  # Add padding and make it expand horizontally
        
        # Add developer credits at the bottom
        credits_frame = ttk.Frame(root)
        credits_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        credits_label = ttk.Label(
            credits_frame, 
            text="Designed and developed by COALA", 
            font=("Helvetica", 9),
            foreground="gray"
        )
        credits_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Wait for user selection
        root.mainloop()

    def on_mode_selected(self, root):
        """Handle mode selection and initialize main UI"""
        # Get the value from the StringVar
        selected_mode = self.editor_mode.get()
        print(f"Selected mode: {selected_mode}")
        
        # Store the mode as a regular string, not a StringVar
        self.editor_mode = selected_mode
        
        # Close the mode selection window
        root.destroy()
        
        # Initialize the main UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize user interface with improved design"""
        # Create main window and set style
        self.root = tk.Tk()
        self.root.title(f"Interactive RMOT Annotation Editor - {self.editor_mode.upper()} Mode")
        self.root.geometry("1600x900")
        
        # Set global theme colors
        style = ttk.Style()
        style.configure(".", font=self.normal_font)
        style.configure("TFrame", background=self.background_color)
        style.configure("TLabel", background=self.background_color, foreground=self.text_color)
        style.configure("TButton", font=self.normal_font, background=self.theme_color)
        style.configure("Accent.TButton", background=self.accent_color)
        style.map("TButton", 
            background=[("active", self.secondary_color), ("pressed", self.secondary_color)],
            foreground=[("active", "white"), ("pressed", "white")])
        
        # Title and header styles
        style.configure("Title.TLabel", font=self.title_font, foreground=self.theme_color)
        style.configure("Subtitle.TLabel", font=self.subtitle_font, foreground=self.secondary_color)
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add title bar
        header_frame = ttk.Frame(main_frame, padding=10, style="Header.TFrame")
        header_frame.pack(fill=tk.X)
        
        header_title = ttk.Label(
            header_frame, 
            text=f"Interactive RMOT Annotation Editor", 
            style="Title.TLabel"
        )
        header_title.pack(side=tk.LEFT, padx=10)
        
        # Mode indicator
        mode_indicator = ttk.Label(
            header_frame, 
            text=f"Mode: {self.editor_mode.upper()}", 
            style="Subtitle.TLabel",
            foreground=self.accent_color
        )
        mode_indicator.pack(side=tk.RIGHT, padx=10)
        
        # Create left image display frame
        self.left_frame = ttk.Frame(main_frame, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create canvas container frame - add border and shadow effects
        canvas_container = ttk.Frame(self.left_frame, padding=2)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        # Fixed display area dimensions
        self.display_width = 1280
        self.display_height = 720
        
        # Image display canvas - use dark background to make images stand out
        self.canvas = tk.Canvas(
            canvas_container, 
            bg="#1E1E1E", 
            width=self.display_width, 
            height=self.display_height,
            highlightthickness=1,
            highlightbackground="#BDBDBD"
        )
        self.canvas.pack(fill=tk.NONE, expand=False)
        
        # Bind mouse events and keyboard events
        # Bind mouse events to canvas
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Double-Button-1>", self.on_left_double_click)
        self.canvas.bind("<Double-Button-3>", self.on_right_double_click)
        
        # Bind zoom and pan events
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom)  # Windows
        self.canvas.bind("<Control-Button-4>", self.on_zoom)    # Linux scroll up
        self.canvas.bind("<Control-Button-5>", self.on_zoom)    # Linux scroll down
        self.canvas.bind("<Button-2>", self.start_pan)          # Middle button
        self.canvas.bind("<B2-Motion>", self.pan)               # Middle button drag
        self.canvas.bind("<ButtonRelease-2>", self.stop_pan)    # Middle button release
        
        # Create bottom control area - use more unified design
        bottom_frame = ttk.Frame(self.left_frame, padding=(0, 10, 0, 0))
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add auto-play controls to bottom control area (defined after bottom_frame, but packed before it)
        auto_play_frame = ttk.LabelFrame(
            self.left_frame, 
            text="Autoplay Control", 
            padding=10,
            borderwidth=1,
            relief="solid"
        )
        auto_play_frame.pack(side=tk.BOTTOM, fill=tk.X, before=bottom_frame, pady=(0, 10))
        
        # Auto-play status display
        self.auto_play_status_label = ttk.Label(
            auto_play_frame, 
            text="Status: Paused", 
            font=self.normal_font
        )
        self.auto_play_status_label.pack(side=tk.LEFT, padx=10)
        
        # Speed control label
        speed_label = ttk.Label(
            auto_play_frame, 
            text="Playback Speed:", 
            font=self.normal_font
        )
        speed_label.pack(side=tk.LEFT, padx=(20, 5))
        
        # Speed adjustment slider
        self.speed_slider = ttk.Scale(
            auto_play_frame,
            from_=1,    # Fastest 1ms
            to=100,     # Slowest 100ms
            orient=tk.HORIZONTAL,
            value=self.auto_play_speed,
            length=150,
            command=self.change_play_speed
        )
        self.speed_slider.pack(side=tk.LEFT)
        
        # Display current speed value
        self.speed_value_label = ttk.Label(
            auto_play_frame,
            text=f"{self.auto_play_speed}ms",
            width=6
        )
        self.speed_value_label.pack(side=tk.LEFT, padx=5)
        
        # Auto-play button
        play_btn = ttk.Button(
            auto_play_frame, 
            text="Play/Pause", 
            command=self.toggle_auto_play,
            style="Accent.TButton",
            width=15
        )
        play_btn.pack(side=tk.RIGHT, padx=10)
        
        # Zoom control group - use card-style design
        zoom_frame = ttk.LabelFrame(
            bottom_frame, 
            text="Zoom Controls", 
            padding=10,
            borderwidth=1,
            relief="solid"
        )
        zoom_frame.pack(fill=tk.X, pady=(0, 10))
        
        zoom_info = ttk.Label(
            zoom_frame, 
            text="Use Ctrl+Scroll to zoom, Middle-click drag to pan", 
            font=self.small_font
        )
        zoom_info.pack(side=tk.LEFT, fill=tk.Y)
        
        self.zoom_label = ttk.Label(zoom_frame, text="Zoom: 100%", width=10)
        self.zoom_label.pack(side=tk.LEFT, padx=10)
        
        reset_zoom_btn = ttk.Button(
            zoom_frame, 
            text="Reset Zoom", 
            command=self.reset_zoom,
            style="Accent.TButton",
            width=15
        )
        reset_zoom_btn.pack(side=tk.RIGHT)
        
        # Navigation controls - centered button group
        nav_frame = ttk.LabelFrame(
            bottom_frame, 
            text="Navigation", 
            padding=10,
            borderwidth=1,
            relief="solid"
        )
        nav_frame.pack(fill=tk.X)
        
        # Create centered button container
        buttons_container = ttk.Frame(nav_frame)
        buttons_container.pack(anchor=tk.CENTER)
        
        # Navigation buttons - use same width and consistent style
        first_frame_btn = ttk.Button(
            buttons_container, 
            text="⏮ First", 
            command=self.go_to_first_frame, 
            width=12
        )
        first_frame_btn.pack(side=tk.LEFT, padx=5)
        
        prev_btn = ttk.Button(
            buttons_container, 
            text="◀ Previous", 
            command=self.prev_frame, 
            width=12
        )
        prev_btn.pack(side=tk.LEFT, padx=5)
        
        next_btn = ttk.Button(
            buttons_container, 
            text="Next ▶", 
            command=self.next_frame, 
            width=12
        )
        next_btn.pack(side=tk.LEFT, padx=5)
        
        jump_frame_btn = ttk.Button(
            buttons_container, 
            text="Jump to...", 
            command=self.jump_to_frame, 
            width=12
        )
        jump_frame_btn.pack(side=tk.LEFT, padx=5)
        
        # Right control panel - use card-style design
        right_frame = ttk.Frame(main_frame, width=320, padding=15)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)  # Prevent frame shrinking
        
        # Dataset selection controls
        folder_frame = ttk.LabelFrame(
            right_frame, 
            text="Dataset", 
            padding=10,
            borderwidth=1,
            relief="solid"
        )
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        open_btn = ttk.Button(
            folder_frame, 
            text="Open Image Directory", 
            command=self.open_directory,
            style="Accent.TButton"
        )
        open_btn.pack(fill=tk.X, pady=5)
        
        self.dataset_label = ttk.Label(folder_frame, text="No dataset loaded", font=self.normal_font)
        self.dataset_label.pack(fill=tk.X)
        
        # Frame information
        self.info_frame = ttk.LabelFrame(
            right_frame, 
            text="Frame Information", 
            padding=10,
            borderwidth=1,
            relief="solid"
        )
        self.info_frame.pack(fill=tk.X, pady=10)
        
        self.frame_info_label = ttk.Label(self.info_frame, text="", font=self.normal_font)
        self.frame_info_label.pack(fill=tk.X)
        
        self.object_count_label = ttk.Label(self.info_frame, text="", font=self.normal_font)
        self.object_count_label.pack(fill=tk.X, pady=(5, 0))
        
        # Display mode toggle
        mode_frame = ttk.LabelFrame(
            right_frame, 
            text="Display Options", 
            padding=10,
            borderwidth=1,
            relief="solid"
        )
        mode_frame.pack(fill=tk.X, pady=10)
        
        self.show_zero_score_var = tk.BooleanVar(value=self.show_zero_score)
        show_zero_score_checkbox = ttk.Checkbutton(
            mode_frame, 
            text="Score=0 Objects",
            variable=self.show_zero_score_var,
            command=self.toggle_zero_score
        )
        show_zero_score_checkbox.pack(fill=tk.X)

        self.show_labels_var = tk.BooleanVar(value=self.show_labels)
        show_labels_checkbox = ttk.Checkbutton(
            mode_frame, 
            text="ID & Score",
            variable=self.show_labels_var,
            command=self.toggle_labels
        )
        show_labels_checkbox.pack(fill=tk.X)
        
        # Merge mode specific controls
        if self.editor_mode == "merge":
            self.merge_frame = ttk.LabelFrame(
                right_frame, 
                text="Merge Controls", 
                padding=10,
                borderwidth=1,
                relief="solid"
            )
            self.merge_frame.pack(fill=tk.X, pady=10)
            
            self.merge_status_label = ttk.Label(
                self.merge_frame, 
                text="Merge Status: Idle",
                font=self.normal_font
            )
            self.merge_status_label.pack(fill=tk.X)
            
            self.tracking_status_label = ttk.Label(
                self.merge_frame, 
                text="No Auto-tracking",
                font=self.normal_font
            )
            self.tracking_status_label.pack(fill=tk.X, pady=(5, 10))
            
            # Add checkbox to control whether to set source object scores to 0
            self.set_source_score_zero_var = tk.BooleanVar(value=self.set_source_score_zero)
            set_source_score_checkbox = ttk.Checkbutton(
                self.merge_frame, 
                text="Set Source Objects' Score to 0",
                variable=self.set_source_score_zero_var,
                command=self.toggle_source_score_zero
            )
            set_source_score_checkbox.pack(fill=tk.X, pady=(0, 10))
            
            merge_buttons_frame = ttk.Frame(self.merge_frame)
            merge_buttons_frame.pack(fill=tk.X)
            
            self.start_merge_btn = ttk.Button(
                merge_buttons_frame, 
                text="Start Merge Mode", 
                command=self.start_merge,
                style="Accent.TButton"
            )
            self.start_merge_btn.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
            
            self.cancel_merge_btn = ttk.Button(
                merge_buttons_frame, 
                text="Cancel All Tracking", 
                command=self.stop_all_tracking, 
                state=tk.DISABLED
            )
            self.cancel_merge_btn.pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True)
        
        # Current frame object list
        object_frame = ttk.LabelFrame(
            right_frame, 
            text="Current Frame Objects", 
            padding=10,
            borderwidth=1,
            relief="solid"
        )
        object_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Use more modern tree view style
        style.configure("Treeview", 
                        background="#FFFFFF",
                        foreground=self.text_color,
                        rowheight=25,
                        fieldbackground="#FFFFFF")
        style.map('Treeview', 
                background=[('selected', self.theme_color)],
                foreground=[('selected', '#FFFFFF')])
        
        # Tree view
        self.object_tree = ttk.Treeview(
            object_frame,
            columns=("ID", "Score", "Info"),
            show="headings",
            selectmode="browse",
            height=6
        )
        
        # Configure headers and columns
        for col, width, text in [("ID", 70, "Object ID"), 
                            ("Score", 70, "Score"), 
                            ("Info", 150, "Information")]:
            self.object_tree.heading(col, text=text)
            self.object_tree.column(col, width=width)
        
        self.object_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        object_scrollbar = ttk.Scrollbar(object_frame, orient="vertical", command=self.object_tree.yview)
        object_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.object_tree.configure(yscrollcommand=object_scrollbar.set)
        
        # Object operation buttons
        button_frame = ttk.Frame(object_frame, padding=(0, 10, 0, 0))
        button_frame.pack(fill=tk.X)
        
        delete_btn = ttk.Button(button_frame, text="Remove Selected", command=self.remove_selected_object)
        delete_btn.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        clear_btn = ttk.Button(button_frame, text="Clear All", command=self.clear_selected_objects)
        clear_btn.pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True)
        
        # JSON mode specific description input
        if self.editor_mode == "json":
            sentence_frame = ttk.LabelFrame(
                right_frame, 
                text="Description", 
                padding=10,
                borderwidth=1,
                relief="solid"
            )
            sentence_frame.pack(fill=tk.X, pady=(10, 10))
            
            ttk.Label(sentence_frame, text="Enter a description:").pack(anchor=tk.W)
            
            self.sentence_entry = ttk.Entry(sentence_frame)
            self.sentence_entry.pack(fill=tk.X, pady=5)
        else:
            # Create an empty entry for merge mode to avoid attribute errors
            self.sentence_entry = ttk.Entry(self.root)
        
        # Save controls
        save_frame = ttk.Frame(right_frame, padding=(0, 0, 0, 10))
        save_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Show relevant buttons based on mode
        if self.editor_mode == "json":
            save_btn = ttk.Button(
                save_frame, 
                text="Save Expressions", 
                command=self.save_annotations,
                style="Accent.TButton"
            )
            save_btn.pack(fill=tk.X, pady=2)
        elif self.editor_mode == "merge":
            save_merged_btn = ttk.Button(
                save_frame, 
                text="Save Merged Boxes", 
                command=self.save_merged_boxes,
                style="Accent.TButton"
            )
            save_merged_btn.pack(fill=tk.X, pady=2)
        
        # Add keyboard shortcuts info card
        keys_frame = ttk.LabelFrame(
            right_frame, 
            text="Keyboard Shortcuts", 
            padding=10,
            borderwidth=1,
            relief="solid"
        )
        keys_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 10))
        
        shortcuts = [
            ("Left Arrow", "Previous Frame"),
            ("Right Arrow", "Next Frame"),
            ("Home", "First Frame"),
            ("Space", "Play/Pause"),
            ("Escape", "Exit Application")
        ]
        
        for key, desc in shortcuts:
            key_frame = ttk.Frame(keys_frame)
            key_frame.pack(fill=tk.X, pady=2)
            
            key_label = ttk.Label(key_frame, text=key, width=12, font=self.small_font)
            key_label.pack(side=tk.LEFT)
            
            desc_label = ttk.Label(key_frame, text=desc, font=self.small_font)
            desc_label.pack(side=tk.LEFT, padx=5)
        
        # Keyboard shortcut bindings
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<Home>", lambda e: self.go_to_first_frame())
        self.root.bind("<space>", lambda e: self.toggle_auto_play())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        
        # Set window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def toggle_auto_play(self):
        """Toggle auto-play status"""
        if self.is_auto_playing:
            self.stop_auto_play()
        else:
            self.start_auto_play()

    def start_auto_play(self):
        """Start auto-play"""
        if not self.is_auto_playing and self.image_files:
            self.is_auto_playing = True
            
            # Update UI to show current status
            if hasattr(self, 'auto_play_status_label'):
                self.auto_play_status_label.config(text="Status: Playing")
            
            # Start timer for auto-play
            self.play_next_frame()

    def stop_auto_play(self):
        """Stop auto-play"""
        if self.is_auto_playing:
            self.is_auto_playing = False
            
            # Update UI to show current status
            if hasattr(self, 'auto_play_status_label'):
                self.auto_play_status_label.config(text="Status: Paused")
            
            # Cancel timer task
            if self.auto_play_job is not None:
                self.root.after_cancel(self.auto_play_job)
                self.auto_play_job = None

    def play_next_frame(self):
        """Play next frame and set timer to continue playing"""
        if not self.is_auto_playing:
            return
            
        # If there's a next frame, switch to next frame
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.update_display()
            
            # Set timer to play next frame
            self.auto_play_job = self.root.after(self.auto_play_speed, self.play_next_frame)
        else:
            # If reached last frame, stop auto-play
            self.stop_auto_play()
            
            # Show playback complete message
            messagebox.showinfo("Autoplay", "Played to the last frame")

    def change_play_speed(self, value):
        """Change auto-play speed"""
        self.auto_play_speed = int(float(value))
        self.speed_value_label.config(text=f"{self.auto_play_speed}ms")
        
        # If playing, reset timer
        if self.is_auto_playing and self.auto_play_job is not None:
            self.root.after_cancel(self.auto_play_job)
            self.auto_play_job = self.root.after(self.auto_play_speed, self.play_next_frame) 
    
    # Add method to toggle source object score setting
    def toggle_source_score_zero(self):
        """Toggle whether to set source objects' score to 0 when merging"""
        self.set_source_score_zero = self.set_source_score_zero_var.get()
    def on_zoom(self, event):
        """Handle zoom events"""
        # Get the current pointer position
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Determine zoom direction
        if event.delta > 0 or event.num == 4:  # Zoom in
            self.zoom_scale *= 1.1
        else:  # Zoom out
            self.zoom_scale /= 1.1
        
        # Limit the zoom range
        self.zoom_scale = max(0.1, min(5.0, self.zoom_scale))
        
        # Update display
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_scale * 100)}%")
        self.update_display()
        
        return "break"  # Prevent default scrolling
    
    def start_pan(self, event):
        """Start panning the image"""
        self.is_panning = True
        self.zoom_start_x = event.x
        self.zoom_start_y = event.y
    
    def pan(self, event):
        """Pan the image"""
        if not self.is_panning:
            return
        
        # Calculate the difference
        dx = (event.x - self.zoom_start_x) / self.zoom_scale
        dy = (event.y - self.zoom_start_y) / self.zoom_scale
        
        # Update pan position
        self.zoom_x -= dx
        self.zoom_y -= dy
        
        # Update starting position
        self.zoom_start_x = event.x
        self.zoom_start_y = event.y
        
        # Redraw
        self.update_display()
    
    def stop_pan(self, event):
        """Stop panning the image"""
        self.is_panning = False
    
    def reset_zoom(self):
        """Reset zoom to default"""
        self.zoom_scale = 1.0
        self.zoom_x = 0
        self.zoom_y = 0
        self.zoom_label.config(text="Zoom: 100%")
        self.update_display()
    
    def toggle_zero_score(self):
        """Toggle display of score=0 objects"""
        self.show_zero_score = self.show_zero_score_var.get()
        
        # In merge mode, force show zero score objects
        if self.editor_mode == "merge" and self.is_merging and not self.show_zero_score:
            self.show_zero_score = True
            self.show_zero_score_var.set(True)
            messagebox.showinfo("Display Mode", "Score=0 objects are always shown in merge mode")
        
        self.update_display()
    def toggle_labels(self):
        """Toggle display of object labels"""
        self.show_labels = self.show_labels_var.get()
        self.update_display()
    def open_directory(self):
        """Open image directory and find corresponding annotation file"""
        # Use fixed dataset directory as initial directory
        initial_dir = self.dataset_default_dir or os.getcwd()
        img_dir = filedialog.askdirectory(title="Select Image Directory", initialdir=initial_dir)
        if not img_dir:
            return
        
        # Save selected directory as default directory for next time
        self.dataset_default_dir = os.path.dirname(img_dir)
        
        self.img_dir = img_dir
        
        # Extract video name
        video_name = os.path.basename(img_dir)
        self.output_data["video_name"] = video_name
        
        # Try to find the annotations file in the parent's "annotations" directory
        parent_dir = os.path.dirname(os.path.dirname(img_dir))
        possible_ann_file = os.path.join(parent_dir, "annotations", f"{video_name}.txt")
        
        if os.path.exists(possible_ann_file):
            self.ann_file = possible_ann_file
            self.load_dataset()
        else:
            # If not found, ask user to select annotation file
            self.ann_file = filedialog.askopenfilename(
                title="Select Annotation File",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if self.ann_file:
                self.load_dataset()
    
    def load_dataset(self):
        """Load dataset and prepare for editing"""
        if not self.img_dir or not self.ann_file:
            return
        
        # Load annotations
        self.annotations = self.read_mot_annotations(self.ann_file)
        
        # Get image files
        self.image_files = self.get_image_files(self.img_dir)
        if not self.image_files:
            messagebox.showerror("Error", f"No images found in directory {self.img_dir}")
            return
        
        # Reset variables
        self.current_idx = 0
        self.persistent_selections = {}
        self.selection_histories = {}
        
        # Reset merge state
        self.is_merging = False
        self.first_merge_box = None
        self.merged_boxes = []
        self.merged_source_ids = set()
        self.auto_tracking_merge = False
        self.tracking_source_ids = []
        self.tracking_start_frame = 0
        self.tracking_merged_id = None
        
        # Reset zoom
        self.zoom_scale = 1.0
        self.zoom_x = 0
        self.zoom_y = 0
        self.zoom_label.config(text="Zoom: 100%")
        
        # Find maximum ID in annotation file
        self.max_id = 0
        for frame_objects in self.annotations.values():
            for obj in frame_objects:
                self.max_id = max(self.max_id, obj['id'])
        
        print(f"Found max ID in annotation file: {self.max_id}")
        
        # Reset output data
        video_name = os.path.basename(self.img_dir)
        self.output_data = {
            "label": {},
            "ignore": {},
            "video_name": video_name,
            "sentence": ""
        }
        
        # Update UI
        self.dataset_label.config(text=f"Dataset: {video_name}")
        
        # Show first frame
        self.update_display()
    
    def read_mot_annotations(self, filepath):
        """Read MOT format annotation file"""
        annotations = defaultdict(list)
        
        with open(filepath, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                if len(data) < 7:  # Ensure we have at least score field
                    continue
                    
                frame_idx = int(data[0])
                track_id = int(data[1])
                bbox_left = int(data[2])
                bbox_top = int(data[3])
                bbox_width = int(data[4])
                bbox_height = int(data[5])
                score = int(data[6])
                
                # Get other fields if they exist
                object_category = int(data[7]) if len(data) > 7 else 0
                truncation = int(data[8]) if len(data) > 8 else 0
                occlusion = int(data[9]) if len(data) > 9 else 0
                
                annotations[frame_idx].append({
                    'id': track_id,
                    'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                    'score': score,
                    'category': object_category,
                    'truncation': truncation,
                    'occlusion': occlusion
                })
        
        return annotations
    
    def get_image_files(self, directory):
        """Get all image files in the directory"""
        image_files = []
        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(directory, filename))
        return image_files
    
    # 1. Add the jump_to_frame method in the InteractiveMOTEditor class

    def jump_to_frame(self):
        """Jump to a specific frame by entering its number"""
        if not self.image_files:
            messagebox.showinfo("Error", "No dataset loaded")
            return
            
        # Create a dialog to input frame number
        frame_dialog = tk.Toplevel(self.root)
        frame_dialog.title("Jump to Frame")
        frame_dialog.geometry("300x150")
        frame_dialog.transient(self.root)  # Set as transient to main window
        frame_dialog.grab_set()  # Make dialog modal
        
        # Center the dialog
        frame_dialog.geometry("+%d+%d" % (
            self.root.winfo_rootx() + self.root.winfo_width()//2 - 150,
            self.root.winfo_rooty() + self.root.winfo_height()//2 - 75))
        
        # Add padding
        content_frame = ttk.Frame(frame_dialog, padding=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Label
        ttk.Label(content_frame, text="Enter frame number:").pack(pady=(0, 10))
        
        # Entry field
        frame_entry = ttk.Entry(content_frame, width=15)
        frame_entry.pack(pady=(0, 15))
        frame_entry.focus_set()  # Set focus to entry field
        
        # Function to handle jump
        def do_jump():
            try:
                frame_num = int(frame_entry.get())
                
                # Find the closest frame in the dataset
                closest_idx = None
                closest_diff = float('inf')
                
                for idx, img_path in enumerate(self.image_files):
                    img_frame_id = self.extract_frame_id(img_path)
                    diff = abs(img_frame_id - frame_num)
                    
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_idx = idx
                
                if closest_idx is not None:
                    self.current_idx = closest_idx
                    self.update_display()
                    frame_dialog.destroy()
                    
                    img_path = self.image_files[self.current_idx]
                    actual_frame_id = self.extract_frame_id(img_path)
                    
                    if actual_frame_id != frame_num:
                        messagebox.showinfo("Frame Jump", 
                                        f"Frame {frame_num} not found. Jumped to closest frame: {actual_frame_id}")
                else:
                    messagebox.showerror("Error", "Could not find a suitable frame")
                    
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid frame number")
        
        # Buttons
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Jump", command=do_jump, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Cancel", command=frame_dialog.destroy).pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True)
        
        # Bind Enter key to the do_jump function
        frame_entry.bind("<Return>", lambda e: do_jump())
        
        # Wait for the dialog to close
        frame_dialog.wait_window()
    def go_to_first_frame(self):
        """Jump to the first frame of the sequence"""
        if not self.image_files:
            return
            
        # Set current index to 0 (first frame)
        self.current_idx = 0
        
        # Update display
        self.update_display()
        
        # Show information message
        self.frame_info_label.config(text=f"Jump to the first frame : {os.path.basename(self.image_files[0])}")

    def extract_frame_id(self, filename):
        """
        Extract the numeric part from filenames like 'img000001.jpg'
        Returns the numeric part as an integer
        """
        import re
        # Remove file extension
        base_name = os.path.splitext(os.path.basename(filename))[0]
        # Find all numeric parts in the filename
        digits = re.findall(r'\d+', base_name)
        if digits:
            # Use the first group of digits found
            return int(digits[0])
        else:
            # If no numeric part is found, use default value
            print(f"Warning: No numeric part found in filename {base_name}")
            return 0  # Default value
    def update_display(self):
        """Update the display with the current frame"""
        if not self.image_files or self.current_idx >= len(self.image_files):
            return
        
        # Read image
        image_path = self.image_files[self.current_idx]
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert before processing
        if frame is None:
            print(f"Cannot read image: {image_path}")
            self.current_idx = min(self.current_idx + 1, len(self.image_files) - 1)
            self.update_display()
            return
        
        # Convert BGR to RGB before processing for correct colors
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get current frame number
        #frame_id = int(os.path.splitext(os.path.basename(image_path))[0])
        frame_id = self.extract_frame_id(image_path)
        self.current_frame_id = frame_id
        
        # Get annotations for current frame
        frame_annotations = self.annotations.get(frame_id, [])
        
        # Count objects with score=0
        zero_score_count = sum(1 for obj in frame_annotations if obj['score'] == 0)
        total_objects = len(frame_annotations)
        
        # Update frame info
        self.frame_info_label.config(text=f"Frame: {frame_id} | Image: {os.path.basename(image_path)}")
        self.object_count_label.config(text=f"Objects: {total_objects} | Score=0: {zero_score_count}")
        
        # Clear current frame bboxes
        self.current_frame_bboxes = []
        
        # Process auto-tracking merge if enabled (only in merge mode)
        if self.editor_mode == "merge" and self.auto_tracking_merge and frame_id > self.tracking_start_frame:
            self.process_auto_tracking()
        
        # Get merged boxes for this frame (only in merge mode)
        merged_boxes_in_current_frame = []
        if self.editor_mode == "merge":
            merged_boxes_in_current_frame = [box for box in self.merged_boxes if box['frame'] == frame_id]
        
        # Create a copy of the original frame for drawing
        display_frame = frame.copy()
        
        # Process annotations and drawing code...
        # [This contains all previous code for processing bounding boxes and labels, unchanged]
        
        # If in merge mode, draw merged boxes first (so they appear under other boxes)
        if self.editor_mode == "merge":
            for merged_box in merged_boxes_in_current_frame:
                x, y, w, h = map(int, merged_box['bbox'])
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), self.merged_color, 2)
                
                # Draw label
                label = f"M:{merged_box['id']}"
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
                cv2.rectangle(display_frame, (x, y - text_size[1] - 2), (x + text_size[0], y), self.merged_color, -1)
                # Use black text for label
                cv2.putText(display_frame, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                        self.font_scale, (0, 0, 0), self.font_thickness)
        
        # If in merge mode and merge process started, draw first merge box
        if self.editor_mode == "merge" and self.is_merging and self.first_merge_box:
            x, y, w, h = map(int, self.first_merge_box['bbox'])
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), self.merging_color, 2)
            
            # Draw label
            label = "Merging..."
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
            cv2.rectangle(display_frame, (x, y - text_size[1] - 2), (x + text_size[0], y), self.merging_color, -1)
            # Use black text for label
            cv2.putText(display_frame, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                    self.font_scale, (0, 0, 0), self.font_thickness)
        
        # Draw bounding boxes
        for obj in frame_annotations:
            # Skip score=0 objects if not showing them and not in merge mode with merging active
            if obj['score'] == 0 and not self.show_zero_score and not (self.editor_mode == "merge" and self.is_merging):
                continue
                
            # Parse bounding box
            x, y, w, h = map(int, obj['bbox'])
            track_id = obj['id']
            score = obj['score']
            
            # Store bbox for interaction
            self.current_frame_bboxes.append({
                'id': track_id,
                'bbox': [x, y, w, h],
                'score': score,
                'category': obj.get('category', 0),
                'truncation': obj.get('truncation', 0),
                'occlusion': obj.get('occlusion', 0),
                'frame': frame_id
            })
            
            # Check if object is persistently selected or in selection history
            is_persistently_selected = track_id in self.persistent_selections
            is_in_history = (track_id in self.selection_histories and 
                            self.current_frame_id in self.selection_histories[track_id])
            
            # Check if object is being tracked for auto-merge (only in merge mode)
            is_tracking_source = self.editor_mode == "merge" and self.auto_tracking_merge and track_id in self.tracking_source_ids

            # Check if object is selected: only mark as selected in and after the selection frame
            is_persistently_selected = track_id in self.persistent_selections and self.current_frame_id >= self.persistent_selections[track_id]
            is_in_history = (track_id in self.selection_histories and 
                            self.current_frame_id in self.selection_histories[track_id])
        
        
            
            # Choose color based on selection status and score
            if is_tracking_source:
                color = self.merging_color  # Yellow for tracked sources
                thickness = 2
            elif is_persistently_selected or is_in_history:
                color = self.selected_color
                thickness = 2
            elif score == 0:
                color = self.zero_score_color
                thickness = 1
            else:
                color = self.normal_color
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Create label with ID and score
            label = f"ID:{track_id} S:{int(score)}"  # Use integer score
            
            
            # Add special marker for selected objects
            if is_tracking_source:
                label = "tracking " + label
            elif is_persistently_selected or is_in_history:
                label = "select " + label
            
            # Only draw labels when show_labels is True
            if self.show_labels:
                # Draw label background
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
                cv2.rectangle(display_frame, (x, y - text_size[1] - 2), (x + text_size[0], y), color, -1)
                
                # Draw label text - using black text color
                cv2.putText(display_frame, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                        self.font_scale, (0, 0, 0), self.font_thickness)
        
        # Display image information
        persistent_selected_count = sum(1 for obj in self.current_frame_bboxes if obj['id'] in self.persistent_selections)
        historical_selected_count = sum(1 for obj in self.current_frame_bboxes 
                                    if obj['id'] in self.selection_histories and 
                                        self.current_frame_id in self.selection_histories[obj['id']] and
                                        obj['id'] not in self.persistent_selections)
        total_selected = persistent_selected_count + historical_selected_count
        
        merged_count = len(merged_boxes_in_current_frame) if self.editor_mode == "merge" else 0
        
        # Modify information text style on the image - use more professional style
        info_text = f"Frame: {frame_id} | Objects: {total_objects} | Selected: {total_selected}"
        if self.editor_mode == "merge":
            info_text += f" | Merged: {merged_count}"
        
        # Use semi-transparent background to improve text readability
        text_background = display_frame.copy()
        overlay = np.zeros((40, display_frame.shape[1], 3), dtype=np.uint8)
        cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 40), (0, 0, 0), -1)
        alpha = 0.6  # Transparency
        cv2.addWeighted(overlay, alpha, text_background[0:40, 0:display_frame.shape[1]], 1 - alpha, 0, text_background[0:40, 0:display_frame.shape[1]])
        display_frame[0:40, 0:display_frame.shape[1]] = text_background[0:40, 0:display_frame.shape[1]]
        
        # Use more aesthetic text style
        cv2.putText(display_frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Bottom control information also uses the same semi-transparent background
        if self.editor_mode == "merge" and self.auto_tracking_merge:
            controls_text = "Auto-tracking IDs: " + str(self.tracking_source_ids) + " | Right-Double-click: Stop tracking"
        elif self.editor_mode == "merge" and self.is_merging:
            controls_text = "Double-click: Select box for merging"
        else:
            controls_text = "Left-click: Select | Right-click: Deselect" 
            if self.editor_mode == "merge":
                controls_text += " | Double-click: Start merge"
        
        # Bottom text background
        text_background = display_frame.copy()
        overlay = np.zeros((40, display_frame.shape[1], 3), dtype=np.uint8)
        bottom_y = display_frame.shape[0] - 40
        cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 40), (0, 0, 0), -1)
        alpha = 0.6  # Transparency
        cv2.addWeighted(overlay, alpha, text_background[bottom_y:bottom_y+40, 0:display_frame.shape[1]], 1 - alpha, 0, text_background[bottom_y:bottom_y+40, 0:display_frame.shape[1]])
        display_frame[bottom_y:bottom_y+40, 0:display_frame.shape[1]] = text_background[bottom_y:bottom_y+40, 0:display_frame.shape[1]]
        
        # Bottom text
        cv2.putText(display_frame, controls_text, (10, display_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Apply zoom and pan transformations
        if self.zoom_scale != 1.0 or self.zoom_x != 0 or self.zoom_y != 0:
            # Get original dimensions
            h, w = display_frame.shape[:2]
            
            # Calculate new dimensions with zoom
            new_h, new_w = int(h * self.zoom_scale), int(w * self.zoom_scale)
            
            # Resize the image
            zoomed_frame = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA if self.zoom_scale < 1 else cv2.INTER_LINEAR)
            
            # Calculate the center of the original image
            center_x, center_y = w // 2, h // 2
            
            # Calculate visible region after pan
            x_offset = int(center_x * self.zoom_scale - center_x + self.zoom_x * self.zoom_scale)
            y_offset = int(center_y * self.zoom_scale - center_y + self.zoom_y * self.zoom_scale)
            
            # Ensure offsets don't go beyond the zoomed image bounds
            x_offset = max(0, min(x_offset, new_w - w))
            y_offset = max(0, min(y_offset, new_h - h))
            
            # Create a blank frame with original dimensions
            display_frame = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Calculate the visible region in the zoomed image
            visible_x_start = x_offset
            visible_y_start = y_offset
            visible_x_end = min(visible_x_start + w, new_w)
            visible_y_end = min(visible_y_start + h, new_h)
            
            # Calculate where to place the visible region in the display frame
            place_x_start = 0
            place_y_start = 0
            place_x_end = visible_x_end - visible_x_start
            place_y_end = visible_y_end - visible_y_start
            
            # Copy the visible portion of the zoomed image to the display frame
            display_frame[place_y_start:place_y_end, place_x_start:place_x_end] = zoomed_frame[visible_y_start:visible_y_end, visible_x_start:visible_x_end]
        
        # Resize image to fit fixed canvas size
        # Get original image dimensions
        img_height, img_width = display_frame.shape[:2]
        
        # Calculate scaling ratio, maintaining original aspect ratio
        width_ratio = self.display_width / img_width
        height_ratio = self.display_height / img_height
        
        # Choose the smaller ratio to ensure image fits completely in canvas
        # If you want to prioritize filling the display area width, choose width_ratio
        scale_ratio = width_ratio
        
        # Calculate dimensions after scaling
        new_width = int(img_width * scale_ratio)
        new_height = int(img_height * scale_ratio)
        
        # Scale image
        resized_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_AREA if scale_ratio < 1 else cv2.INTER_LINEAR)
        
        # Calculate image position in canvas (centered)
        x_offset = (self.display_width - new_width) // 2
        y_offset = (self.display_height - new_height) // 2
        
        # Create black background with same size as canvas
        canvas_image = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Place scaled image in center of background
        canvas_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
        
        # Now image is already in RGB format, pass directly to tkinter
        img = tk.PhotoImage(data=cv2.imencode('.png', canvas_image)[1].tobytes())
        
        # Store image reference to prevent garbage collection
        self.current_image = img
        
        # Clear canvas and display new image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
        
        # Update object tree
        self.update_object_tree()
        
        # Update output data for current frame
        self.update_output_data()
        
        # Update merge status (in merge mode only)
        if self.editor_mode == "merge":
            if self.auto_tracking_merge:
                tracking_text = f"Auto-tracking IDs {self.tracking_source_ids} since frame {self.tracking_start_frame}"
                self.tracking_status_label.config(text=tracking_text)
                self.merge_status_label.config(text=f"Merged ID: {self.tracking_merged_id} | Current frame: {frame_id}")
                
                # Make sure cancel button is enabled
                self.cancel_merge_btn.config(state=tk.NORMAL)
                self.start_merge_btn.config(state=tk.DISABLED)
            elif self.is_merging:
                if self.first_merge_box:
                    self.merge_status_label.config(text="Merge Status: Select second box")
                else:
                    self.merge_status_label.config(text="Merge Status: Select first box")
                self.tracking_status_label.config(text="No Auto-tracking")
            else:
                merged_count = len(merged_boxes_in_current_frame)
                self.merge_status_label.config(text=f"Merge Status: Idle | Merged boxes: {merged_count}")
                self.tracking_status_label.config(text="No Auto-tracking")
                
                # Make sure score=0 objects are shown in merge mode, hidden in regular mode
                if not self.is_merging and self.show_zero_score:
                    self.show_zero_score = False
                    self.show_zero_score_var.set(False)
                    # No need to call update_display here to avoid recursion
    
    def process_auto_tracking(self):
        """Process automatic tracking for merged bboxes"""
        if not self.auto_tracking_merge or not self.tracking_source_ids:
            return
        
        # Ensure source IDs are in pairs
        if len(self.tracking_source_ids) % 2 != 0:
            print("Warning: tracking_source_ids length is not even")
            return
        
        # Process each pair of source IDs
        for i in range(0, len(self.tracking_source_ids), 2):
            # Ensure there's a pair of IDs to process
            if i + 1 >= len(self.tracking_source_ids):
                continue
                
            source1_id = self.tracking_source_ids[i]
            source2_id = self.tracking_source_ids[i+1]
            
            # Find source objects in current frame
            source1 = None
            source2 = None
            
            for obj in self.annotations.get(self.current_frame_id, []):
                if obj['id'] == source1_id:
                    source1 = obj
                elif obj['id'] == source2_id:
                    source2 = obj
            
            # If both source objects are found, create merged bbox
            if source1 and source2:
                # Calculate merged bbox (union of both boxes)
                bbox1 = source1['bbox']
                bbox2 = source2['bbox']
                
                min_x = min(bbox1[0], bbox2[0])
                min_y = min(bbox1[1], bbox2[1])
                max_x = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
                max_y = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
                
                merged_width = max_x - min_x
                merged_height = max_y - min_y
                
                # Get corresponding merged ID
                merged_id = None
                if isinstance(self.tracking_merged_id, list):
                    if i//2 < len(self.tracking_merged_id):
                        merged_id = self.tracking_merged_id[i//2]
                else:
                    if i == 0:
                        merged_id = self.tracking_merged_id
                
                if merged_id is None:
                    continue
                    
                # Create merged box
                merged_box = {
                    'frame': self.current_frame_id,
                    'id': merged_id,
                    'bbox': [min_x, min_y, merged_width, merged_height],
                    'score': 1,
                    'category': source1['category'],
                    'truncation': 0,
                    'occlusion': 0,
                    'merged_from': [source1_id, source2_id]
                }
                
                # Check if there's already a merged box with same ID and frame
                existing = False
                for box in self.merged_boxes:
                    if (box['frame'] == self.current_frame_id and 
                        box['id'] == merged_id):
                        existing = True
                        break
                
                # If it doesn't exist, add to merged boxes list
                if not existing:
                    self.merged_boxes.append(merged_box)
    
    def update_object_tree(self):
        """Update the list of selected objects in the current frame"""
        # Clear existing items
        for item in self.object_tree.get_children():
            self.object_tree.delete(item)
        
        # Get selected objects in current frame (both persistent and historical)
        selected_objects = []
        for obj in self.current_frame_bboxes:
            # Skip score=0 objects if not showing them and not in merge mode with merging active
            if obj['score'] == 0 and not self.show_zero_score and not (self.editor_mode == "merge" and self.is_merging):
                continue
                
            obj_id = obj['id']
            is_persistently_selected = obj_id in self.persistent_selections and self.current_frame_id >= self.persistent_selections[obj_id]
            is_in_history = (obj_id in self.selection_histories and 
                            self.current_frame_id in self.selection_histories[obj_id])
            
            if is_persistently_selected or is_in_history:
                # Make a copy to avoid modifying original data
                obj_copy = obj.copy()
                obj_copy['selection_type'] = 'Current' if is_persistently_selected else 'History'
                selected_objects.append(obj_copy)
            
            # Add tracking sources with special tag (only in merge mode)
            if self.editor_mode == "merge" and self.auto_tracking_merge and obj_id in self.tracking_source_ids:
                obj_copy = obj.copy()
                obj_copy['selection_type'] = 'Tracking'
                selected_objects.append(obj_copy)
        
        # Add to treeview - first add selected objects
        for obj in selected_objects:
            obj_id = obj['id']
            selection_type = obj['selection_type']
            
            # Get info about when selection started
            if selection_type == 'Current':
                frame_info = f"Frame {self.persistent_selections.get(obj_id, self.current_frame_id)}"
            elif selection_type == 'Tracking':
                frame_info = f"Auto-tracking since {self.tracking_start_frame}"
            else:
                # For historical selections, find the first frame it was selected
                history_frames = self.selection_histories.get(obj_id, [])
                if history_frames:
                    first_frame = min(history_frames)
                    frame_info = f"History ({first_frame})"
                else:
                    frame_info = "History"
            
            self.object_tree.insert(
                "", 
                tk.END, 
                values=(
                    obj_id, 
                    f"{int(obj['score'])}",  # Integer score
                    frame_info
                ),
                tags=(selection_type.lower(),)
            )
        
        # Add merged boxes to tree (only in merge mode)
        if self.editor_mode == "merge":
            merged_boxes_in_frame = [box for box in self.merged_boxes if box['frame'] == self.current_frame_id]
            for merged_box in merged_boxes_in_frame:
                is_tracking = self.auto_tracking_merge and merged_box['id'] == self.tracking_merged_id
                tag = "tracking_merged" if is_tracking else "merged"
                
                info_text = "Auto-tracking" if is_tracking else f"Merged from {merged_box['merged_from']}"
                
                self.object_tree.insert(
                    "",
                    tk.END,
                    values=(
                        f"M:{merged_box['id']}",
                        "1",  # Integer score
                        info_text
                    ),
                    tags=(tag,)
                )
        
        # Set colors for different selection types
        self.object_tree.tag_configure("current", background="#FFD700")       # Gold for current selections
        self.object_tree.tag_configure("history", background="#A0A0A0")       # Gray for historical selections
        
        # Merge-specific tags (only in merge mode)
        if self.editor_mode == "merge":
            self.object_tree.tag_configure("merged", background="#FF00FF")        # Purple for merged boxes
            self.object_tree.tag_configure("tracking", background="#FFFF00")      # Yellow for tracking sources
            self.object_tree.tag_configure("tracking_merged", background="#FF00FF")  # Purple with different text for tracked merges
        
    def update_output_data(self):
        """Update output data based on current selections"""
        # Update for current frame
        frame_key = str(self.current_frame_id)
        
        if frame_key not in self.output_data["label"]:
            self.output_data["label"][frame_key] = []
        
        # Clear existing entries for this frame
        self.output_data["label"][frame_key] = []
        
        # Add currently selected objects
        for obj_id in self.persistent_selections:
            # Check if this object exists in the current frame
            obj_exists = self.object_exists_in_frame(obj_id, self.current_frame_id)
            
            if obj_exists:
                if obj_id not in self.output_data["label"][frame_key]:
                    self.output_data["label"][frame_key].append(obj_id)
    

    def object_exists_in_frame(self, obj_id, frame_idx):
        """Check if the specified object ID exists in the specified frame"""
        # Get all annotated objects for this frame
        frame_objects = self.annotations.get(frame_idx, [])
        
        # Check if the object ID exists in this frame
        for obj in frame_objects:
            if obj['id'] == obj_id:
                return True
                
        return False

    def find_clicked_bbox(self, img_x, img_y):
        """Find all bounding boxes at the clicked position, prioritizing smaller ones"""
        # Store all bounding boxes containing the click position
        candidate_objects = []
        
        for obj in self.current_frame_bboxes:
            bbox = obj['bbox']
            if (bbox[0] <= img_x <= bbox[0] + bbox[2] and 
                bbox[1] <= img_y <= bbox[1] + bbox[3]):
                # Calculate bounding box area
                area = bbox[2] * bbox[3]
                candidate_objects.append((obj, area))
        
        # If there are multiple candidate bounding boxes, sort by area from small to large
        if candidate_objects:
            candidate_objects.sort(key=lambda x: x[1])  # Sort by area
            return candidate_objects[0][0]  # Return the smallest bounding box
        
        return None  # If none found, return None

    def on_left_click(self, event):
        """Handle left click to select objects"""
        if self.editor_mode == "merge" and (self.is_merging or self.auto_tracking_merge):
            return  # Ignore regular left clicks during merge or tracking mode
                
        if not self.current_frame_bboxes:
            return
            
        # Get original image dimensions
        img_path = self.image_files[self.current_idx]
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        # Calculate scaling ratio in display area
        width_ratio = self.display_width / img_width
        height_ratio = self.display_height / img_height
        scale_ratio = width_ratio  # Use same scaling logic as in update_display
        
        # Calculate dimensions after scaling
        new_width = int(img_width * scale_ratio)
        new_height = int(img_height * scale_ratio)
        
        # Calculate image position in canvas
        x_offset = (self.display_width - new_width) // 2
        y_offset = (self.display_height - new_height) // 2
        
        # Get click coordinates in canvas space
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Check if click is within image area
        if (canvas_x < x_offset or canvas_x >= x_offset + new_width or
            canvas_y < y_offset or canvas_y >= y_offset + new_height):
            return  # Click is outside the image
        
        # Convert from canvas coordinates to image coordinates
        img_x = (canvas_x - x_offset) / scale_ratio
        img_y = (canvas_y - y_offset) / scale_ratio
        
        # Apply zoom scale considerations
        if self.zoom_scale != 1.0:
            img_x = img_x * self.zoom_scale + self.zoom_x
            img_y = img_y * self.zoom_scale + self.zoom_y
        
        # Use the new function to find bounding box at click position, prioritizing smaller ones
        clicked_obj = self.find_clicked_bbox(img_x, img_y)
        
        if clicked_obj:
            obj_id = clicked_obj['id']
        
            # Modified: Only add selection in current frame
            if obj_id not in self.persistent_selections:
                # Store current frame as first selection frame
                self.persistent_selections[obj_id] = self.current_frame_id
                
                # Initialize or update selection history
                if obj_id not in self.selection_histories:
                    self.selection_histories[obj_id] = []
                
                # Only add current frame to history, not previous frames
                if self.current_frame_id not in self.selection_histories[obj_id]:
                    self.selection_histories[obj_id].append(self.current_frame_id)
        
            # Update display
            self.update_display()
        
    def on_right_click(self, event):
        """Handle right click to deselect objects"""
        if self.editor_mode == "merge" and (self.is_merging or self.auto_tracking_merge):
            return  # Ignore right clicks during merge or tracking mode
                
        if not self.current_frame_bboxes:
            return
            
        # Get original image dimensions
        img_path = self.image_files[self.current_idx]
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        # Calculate scaling ratio in display area
        width_ratio = self.display_width / img_width
        height_ratio = self.display_height / img_height
        scale_ratio = width_ratio  # Using the same scaling logic as in update_display
        
        # Calculate dimensions after scaling
        new_width = int(img_width * scale_ratio)
        new_height = int(img_height * scale_ratio)
        
        # Calculate image position in canvas
        x_offset = (self.display_width - new_width) // 2
        y_offset = (self.display_height - new_height) // 2
        
        # Get click coordinates in canvas space
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Check if click is within image area
        if (canvas_x < x_offset or canvas_x >= x_offset + new_width or
            canvas_y < y_offset or canvas_y >= y_offset + new_height):
            return  # Click is outside the image area
        
        # Convert from canvas coordinates to image coordinates
        img_x = (canvas_x - x_offset) / scale_ratio
        img_y = (canvas_y - y_offset) / scale_ratio
        
        # Apply zoom scale considerations
        if self.zoom_scale != 1.0:
            img_x = img_x * self.zoom_scale + self.zoom_x
            img_y = img_y * self.zoom_scale + self.zoom_y
        
        # Use new function to find bounding box at click position, prioritizing smaller ones
        clicked_obj = self.find_clicked_bbox(img_x, img_y)
        
        if clicked_obj and clicked_obj['id'] in self.persistent_selections:
            # Continue processing deselection logic
            # Get the object ID
            obj_id = clicked_obj['id']
        
        if clicked_obj and clicked_obj['id'] in self.persistent_selections:
            # Get the object ID
            obj_id = clicked_obj['id']
            
            # Get the starting frame for this object
            start_frame = self.persistent_selections[obj_id]
            
            # Get all frames where this object was selected
            selected_frames = []
            if obj_id in self.selection_histories:
                selected_frames = self.selection_histories[obj_id].copy()
            
            # Remove from persistent selections
            del self.persistent_selections[obj_id]
            
            # Since we're deselecting, update selection histories - keep frames up to current
            # Important: Exclude the current frame since we're deselecting now
            end_frame = self.current_frame_id - 1  # Exclude current frame
            
            # Calculate the range of frames to keep in output
            frames_to_keep = list(range(start_frame, end_frame + 1))
            
            # Merge with existing selection history frames
            for frame in frames_to_keep:
                if frame not in selected_frames:
                    selected_frames.append(frame)
            
            # Filter out any frames that are >= current frame
            selected_frames = [frame for frame in selected_frames if frame < self.current_frame_id]
            
            # Sort the frames
            selected_frames.sort()
            
            # Update selection history with final list
            self.selection_histories[obj_id] = selected_frames
            
            # Update display
            self.update_display()
    
    def on_left_double_click(self, event):
        """Handle left double click to start merge process"""
        # Only available in merge mode
        if self.editor_mode != "merge":
            return
                
        if not self.current_frame_bboxes:
            return
            
        # Get original image dimensions
        img_path = self.image_files[self.current_idx]
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        # Calculate scaling ratio in display area
        width_ratio = self.display_width / img_width
        height_ratio = self.display_height / img_height
        scale_ratio = width_ratio  # Using the same scaling logic as in update_display
        
        # Calculate dimensions after scaling
        new_width = int(img_width * scale_ratio)
        new_height = int(img_height * scale_ratio)
        
        # Calculate image position in canvas
        x_offset = (self.display_width - new_width) // 2
        y_offset = (self.display_height - new_height) // 2
        
        # Get click coordinates in canvas space
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Check if click is within image area
        if (canvas_x < x_offset or canvas_x >= x_offset + new_width or
            canvas_y < y_offset or canvas_y >= y_offset + new_height):
            return  # Click is outside the image area
        
        # Convert from canvas coordinates to image coordinates
        img_x = (canvas_x - x_offset) / scale_ratio
        img_y = (canvas_y - y_offset) / scale_ratio
        
        # Apply zoom scale considerations
        if self.zoom_scale != 1.0:
            img_x = img_x * self.zoom_scale + self.zoom_x
            img_y = img_y * self.zoom_scale + self.zoom_y
        
        # Check if click is within any bounding box
        clicked_obj = None
        for obj in self.current_frame_bboxes:
            bbox = obj['bbox']
            if (bbox[0] <= img_x <= bbox[0] + bbox[2] and 
                bbox[1] <= img_y <= bbox[1] + bbox[3]):
                clicked_obj = obj
                break
        
        if clicked_obj:
            # Allow new merge operations even in auto-tracking state
            if self.is_merging:
                # If already in merge mode, use clicked object as second merge box
                self.complete_merge(clicked_obj)
            else:
                # Otherwise start merge mode
                self.start_merge_with_box(clicked_obj)
        
    def on_right_double_click(self, event):
        """Handle right double click to stop auto-tracking"""
        if self.editor_mode == "merge" and self.auto_tracking_merge:
            # Get original image dimensions
            img_path = self.image_files[self.current_idx]
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]
            
            # Calculate scaling ratio in display area
            width_ratio = self.display_width / img_width
            height_ratio = self.display_height / img_height
            scale_ratio = width_ratio  # Using the same scaling logic as in update_display
            
            # Calculate dimensions after scaling
            new_width = int(img_width * scale_ratio)
            new_height = int(img_height * scale_ratio)
            
            # Calculate image position in canvas
            x_offset = (self.display_width - new_width) // 2
            y_offset = (self.display_height - new_height) // 2
            
            # Get click coordinates in canvas space
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            # Check if click is within image area
            if (canvas_x < x_offset or canvas_x >= x_offset + new_width or
                canvas_y < y_offset or canvas_y >= y_offset + new_height):
                # Click is outside the image area, stop all tracking
                self.stop_all_tracking()
                return
            
            # Convert from canvas coordinates to image coordinates
            img_x = (canvas_x - x_offset) / scale_ratio
            img_y = (canvas_y - y_offset) / scale_ratio
            
            # Apply zoom scale considerations
            if self.zoom_scale != 1.0:
                img_x = img_x * self.zoom_scale + self.zoom_x
                img_y = img_y * self.zoom_scale + self.zoom_y
            
            # First check if a merged box was clicked
            clicked_merged_box = None
            for box in self.merged_boxes:
                if box['frame'] == self.current_frame_id:
                    x, y, w, h = map(int, box['bbox'])
                    if (x <= img_x <= x + w and y <= img_y <= y + h):
                        clicked_merged_box = box
                        break
            
            if clicked_merged_box:
                # If a merged box was clicked, stop tracking for that specific merge
                self.stop_specific_tracking(clicked_merged_box['id'])
            else:
                # If no merged box was clicked, check if a tracked source object was clicked
                clicked_obj = None
                for obj in self.current_frame_bboxes:
                    bbox = obj['bbox']
                    if (bbox[0] <= img_x <= bbox[0] + bbox[2] and 
                        bbox[1] <= img_y <= bbox[1] + bbox[3]):
                        clicked_obj = obj
                        break
                
                if clicked_obj and clicked_obj['id'] in self.tracking_source_ids:
                    # If a tracked source object was clicked, find the corresponding merge and stop tracking
                    idx = self.tracking_source_ids.index(clicked_obj['id'])
                    pair_idx = idx // 2
                    if isinstance(self.tracking_merged_id, list) and pair_idx < len(self.tracking_merged_id):
                        self.stop_specific_tracking(self.tracking_merged_id[pair_idx])
                    elif not isinstance(self.tracking_merged_id, list) and pair_idx == 0:
                        self.stop_specific_tracking(self.tracking_merged_id)
                else:
                    # If no relevant objects were clicked, stop all tracking
                    self.stop_all_tracking()

    def stop_specific_tracking(self, merged_id):
        """Stop tracking of a specific merged box"""
        if not self.auto_tracking_merge:
            return
        
        # Record end frame
        end_frame = self.current_frame_id
        
        # Find corresponding source IDs
        sources_to_remove = []
        if isinstance(self.tracking_merged_id, list):
            # Multiple merge case
            if merged_id in self.tracking_merged_id:
                idx = self.tracking_merged_id.index(merged_id)
                start_idx = idx * 2
                if start_idx + 1 < len(self.tracking_source_ids):
                    sources_to_remove = [self.tracking_source_ids[start_idx], self.tracking_source_ids[start_idx + 1]]
                
                # Remove from tracking list
                self.tracking_merged_id.pop(idx)
                
                # If only one merged ID remains, convert back to single value
                if len(self.tracking_merged_id) == 1:
                    self.tracking_merged_id = self.tracking_merged_id[0]
        else:
            # Single merge case
            if self.tracking_merged_id == merged_id:
                if len(self.tracking_source_ids) >= 2:
                    sources_to_remove = [self.tracking_source_ids[0], self.tracking_source_ids[1]]
                self.tracking_merged_id = None
        
        # Remove from source ID list
        if sources_to_remove:
            self.tracking_source_ids = [id for id in self.tracking_source_ids if id not in sources_to_remove]
        
        # If no more tracking, turn off auto-tracking
        if not self.tracking_source_ids or self.tracking_merged_id is None:
            self.auto_tracking_merge = False
            self.tracking_source_ids = []
            self.tracking_merged_id = None
            
            # Reset UI state
            self.cancel_merge_btn.config(state=tk.DISABLED)
            self.start_merge_btn.config(state=tk.NORMAL)
        
        # Update display
        self.update_display()
        
        messagebox.showinfo("Tracking Stopped", 
                        f"Stopped tracking merged box with ID {merged_id}.")

    def stop_all_tracking(self):
        """Stop all auto-tracking"""
        if not self.auto_tracking_merge:
            return
            
        # Record end frame
        end_frame = self.current_frame_id
        
        # Get number of tracked merges
        if isinstance(self.tracking_merged_id, list):
            merged_count = len(self.tracking_merged_id)
        else:
            merged_count = 1 if self.tracking_merged_id is not None else 0
        
        # Stop all tracking
        self.auto_tracking_merge = False
        self.tracking_source_ids = []
        self.tracking_merged_id = None
        
        # Reset UI state
        self.cancel_merge_btn.config(state=tk.DISABLED)
        self.start_merge_btn.config(state=tk.NORMAL)
        
        # Update display
        self.update_display()
        
        messagebox.showinfo("All Tracking Stopped", 
                        f"Stopped tracking all {merged_count} merged boxes at frame {end_frame}.")
    
    def start_merge(self):
        """Start merge process without a specific first box"""
        # Only available in merge mode
        if self.editor_mode != "merge":
            return
                
        if not self.current_frame_bboxes:
            messagebox.showinfo("Merge", "No objects to merge in current frame")
            return
        
        # Force show zero score objects in merge mode
        if not self.show_zero_score:
            self.show_zero_score = True
            self.show_zero_score_var.set(True)
        
        # Allow new merges while auto-tracking
        self.is_merging = True
        self.first_merge_box = None
        self.merge_status_label.config(text="Merge Status: Select first box")
        self.cancel_merge_btn.config(state=tk.NORMAL)
        
        # Don't disable start merge button, allow restarting merge at any time
        messagebox.showinfo("Merge Mode", "Double-click on the first box to merge")

    def start_merge_with_box(self, first_box):
        """Start merge process with a specific first box"""
        # Only available in merge mode
        if self.editor_mode != "merge":
            return
                
        # Force show zero score objects in merge mode
        if not self.show_zero_score:
            self.show_zero_score = True
            self.show_zero_score_var.set(True)
                
        self.is_merging = True
        self.first_merge_box = first_box
        self.merge_status_label.config(text="Merge Status: Select second box")
        self.cancel_merge_btn.config(state=tk.NORMAL)
        
        # Don't disable merge button, allowing users to start new merges at any time
        # self.start_merge_btn.config(state=tk.DISABLED)  # Remove this line
        
        messagebox.showinfo("Merge Mode", "Double-click on the second box to merge")
        self.update_display()

    def complete_merge(self, second_box):
        """Complete merge process with second box"""
        # Only available in merge mode
        if self.editor_mode != "merge":
            return
                
        if not self.first_merge_box:
            # If first box wasn't set, use the second box as first and wait for another selection
            self.first_merge_box = second_box
            self.merge_status_label.config(text="Merge Status: Select second box")
            self.update_display()
            return
        
        # Calculate merged bbox (union of both boxes)
        first_bbox = self.first_merge_box['bbox']
        second_bbox = second_box['bbox']
        
        min_x = min(first_bbox[0], second_bbox[0])
        min_y = min(first_bbox[1], second_bbox[1])
        max_x = max(first_bbox[0] + first_bbox[2], second_bbox[0] + second_bbox[2])
        max_y = max(first_bbox[1] + first_bbox[3], second_bbox[1] + second_bbox[3])
        
        merged_width = max_x - min_x
        merged_height = max_y - min_y
        
        # Increment max ID for the new merged box
        self.max_id += 1
        new_id = self.max_id
        
        # Create merged box
        merged_box = {
            'frame': self.current_frame_id,
            'id': new_id,
            'bbox': [min_x, min_y, merged_width, merged_height],
            'score': 1,  # Default score for merged box
            'category': self.first_merge_box['category'],  # Use first object's category
            'truncation': 0,
            'occlusion': 0,
            'merged_from': [self.first_merge_box['id'], second_box['id']]
        }
        
        # Add to merged boxes list
        self.merged_boxes.append(merged_box)
        
        # Add source IDs to the set of merged sources
        self.merged_source_ids.add(self.first_merge_box['id'])
        self.merged_source_ids.add(second_box['id'])
        
        # Add new source IDs to tracking list
        if not self.auto_tracking_merge:
            # First merge, initialize tracking state
            self.auto_tracking_merge = True
            self.tracking_source_ids = [self.first_merge_box['id'], second_box['id']]
            self.tracking_start_frame = self.current_frame_id
            self.tracking_merged_id = new_id
        else:
            # Already tracking, add new source IDs to tracking list
            self.tracking_source_ids.extend([self.first_merge_box['id'], second_box['id']])
            # If this is the second pair of source IDs, set tracking_merged_id as a list
            if not isinstance(self.tracking_merged_id, list):
                self.tracking_merged_id = [self.tracking_merged_id, new_id]
            else:
                self.tracking_merged_id.append(new_id)
        
        # Display message
        messagebox.showinfo("Merge Completed",
                        f"Created merged box with ID {new_id}. "
                        f"Now tracking sources {self.tracking_source_ids}. "
                        f"Right-double-click to stop all tracking.")
        
        # Reset merge state to allow starting new merges
        self.is_merging = False
        self.first_merge_box = None
        
        # Update display
        self.update_display()

    def stop_auto_tracking(self):
        """Stop auto-tracking of merged boxes"""
        # Only available in merge mode
        if self.editor_mode != "merge":
            return
                
        if not self.auto_tracking_merge:
            return
                
        # Record end frame for auto-tracking
        end_frame = self.current_frame_id
        
        # Get number of tracked merges
        merged_count = len(self.tracking_source_ids) // 2
        
        # Disable auto-tracking
        self.auto_tracking_merge = False
        self.tracking_source_ids = []
        self.tracking_merged_id = None
        
        # Reset UI state
        self.cancel_merge_btn.config(state=tk.DISABLED)
        self.start_merge_btn.config(state=tk.NORMAL)
        
        # Update display
        self.update_display()
        
        messagebox.showinfo("Auto-tracking Stopped", 
                        f"Auto-tracking stopped at frame {end_frame}. "
                        f"Stopped tracking {merged_count} merged boxes.")

    def remove_selected_object(self):
        """Remove the selected object from the treeview"""
        selected_item = self.object_tree.selection()
        if not selected_item:
            return
        
        # Get object ID from treeview
        obj_id_str = self.object_tree.item(selected_item, 'values')[0]
        
        # Check if it's a merged box
        if obj_id_str.startswith("M:"):
            # Only available in merge mode
            if self.editor_mode != "merge":
                return
                
            # Find and remove the merged box
            merge_id = int(obj_id_str.split(":")[-1])
            for i, box in enumerate(self.merged_boxes):
                if box['id'] == merge_id and box['frame'] == self.current_frame_id:
                    del self.merged_boxes[i]
                    
                    # If tracking this box, stop tracking
                    if self.auto_tracking_merge and self.tracking_merged_id == merge_id:
                        self.stop_auto_tracking()
                    
                    break
            self.update_display()
            return
        
        # Regular selected object
        obj_id = int(obj_id_str)
        
        # If it's a tracking source, stop tracking (only in merge mode)
        if self.editor_mode == "merge" and self.auto_tracking_merge and obj_id in self.tracking_source_ids:
            self.stop_auto_tracking()
        
        # Remove from persistent selections
        if obj_id in self.persistent_selections:
            # Get the starting frame for this object
            start_frame = self.persistent_selections[obj_id]
            
            # Get all frames where this object was selected
            selected_frames = []
            if obj_id in self.selection_histories:
                selected_frames = self.selection_histories[obj_id]
            
            # Remove from persistent selections
            del self.persistent_selections[obj_id]
            
            # Since we're deselecting, update selection histories - keep frames up to current
            end_frame = self.current_frame_id - 1  # Exclude current frame
            
            # Calculate the range of frames to keep in output
            frames_to_keep = range(start_frame, end_frame + 1)
            
            # Add all frames from start to current-1 to selection history
            for frame in frames_to_keep:
                if frame not in selected_frames:
                    selected_frames.append(frame)
            
            # Update selection history with final list
            self.selection_histories[obj_id] = selected_frames
        
        # Update display
        self.update_display()
    
    def clear_selected_objects(self):
        """Clear all selected objects"""
        if not self.persistent_selections:
            return
        
        # Get all persistent selections and preserve their histories
        for obj_id, start_frame in list(self.persistent_selections.items()):
            # Get all frames where this object was selected
            selected_frames = []
            if obj_id in self.selection_histories:
                selected_frames = self.selection_histories[obj_id]
            
            # Calculate the range of frames to keep in output (up to current frame - 1)
            end_frame = self.current_frame_id - 1  # Exclude current frame
            frames_to_keep = range(start_frame, end_frame + 1)
            
            # Filter to keep only frames up to current-1
            frames_to_keep_filtered = [f for f in selected_frames if f <= end_frame]
            
            # Add all frames from start to current-1 to selection history
            for frame in frames_to_keep:
                if frame not in frames_to_keep_filtered:
                    frames_to_keep_filtered.append(frame)
            
            # Update selection history with final list
            self.selection_histories[obj_id] = frames_to_keep_filtered
        
        # Clear persistent selections but keep histories
        self.persistent_selections = {}
        
        # Update display
        self.update_display()
    
    def prev_frame(self):
        """Go to previous frame"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def next_frame(self):
        """Go to next frame"""
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.update_display()

    def save_annotations(self):
        """Save annotations to JSON file"""
        # Only available in JSON mode
        if self.editor_mode != "json":
            messagebox.showinfo("Mode Error", "Annotation saving is only available in JSON mode")
            return
            
        # Update sentence
        self.output_data["sentence"] = self.sentence_entry.get()
        
        # Get description text for filename
        description = self.sentence_entry.get().strip()
        
        # Handle empty description case
        if not description:
            # If no description, use video name as fallback
            file_prefix = self.output_data['video_name']
        else:
            # Clean description text, remove illegal characters to make it suitable as filename
            import re
            # Replace illegal characters with underscores
            clean_desc = re.sub(r'[\\/*?:"<>|]', "_", description)
            """ # Limit length to avoid overly long filenames
            if len(clean_desc) > 150:
                clean_desc = clean_desc[:147] + "..." """
            # Use only description as filename, not including video name
            file_prefix = clean_desc
        
        # Build comprehensive label data from all selection histories
        label_data = {}
        
        # Get annotation information for all frames
        all_frames_annotations = {}
        for frame_idx, objects in self.annotations.items():
            all_frames_annotations[frame_idx] = {obj['id'] for obj in objects}
        
        # Process historical selection records
        for obj_id, frames in self.selection_histories.items():
            for frame_idx in frames:
                # Check if this object ID exists in this frame
                if frame_idx in all_frames_annotations and obj_id in all_frames_annotations[frame_idx]:
                    frame_key = str(frame_idx)
                    if frame_key not in label_data:
                        label_data[frame_key] = []
                    
                    if obj_id not in label_data[frame_key]:
                        label_data[frame_key].append(obj_id)
        
        # Process currently persistent selected objects
        for obj_id in self.persistent_selections:
            # Get first selection frame
            first_frame = self.persistent_selections[obj_id]
            # Get current frame
            last_frame = self.current_frame_id
            
            # Add all frames between first selection and current frame
            for frame_idx in range(first_frame, last_frame + 1):
                # Check if this object ID exists in this frame
                if frame_idx in all_frames_annotations and obj_id in all_frames_annotations[frame_idx]:
                    frame_key = str(frame_idx)
                    if frame_key not in label_data:
                        label_data[frame_key] = []
                    
                    if obj_id not in label_data[frame_key]:
                        label_data[frame_key].append(obj_id)
        
        # Update output data
        self.output_data["label"] = label_data
        
        # Convert keys to strings for JSON serialization
        output_json = {
            "label": {str(k): v for k, v in self.output_data["label"].items()},
            "ignore": {str(k): v for k, v in self.output_data["ignore"].items()},
            "video_name": self.output_data["video_name"],
            "sentence": self.output_data["sentence"]
        }
        
        initial_save_dir = self.save_default_dir
        if not initial_save_dir:
            # If no save directory is set, try to use annotation file directory
            initial_save_dir = os.path.dirname(os.path.join(self.ann_file,"expressions")) if self.ann_file else None
        
        # Ask user for save location, using description as default filename
        save_path = filedialog.asksaveasfilename(
            initialdir=initial_save_dir,
            initialfile=f"{file_prefix}.json",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
        
        # Save to file
        try:
            with open(save_path, 'w') as f:
                json.dump(output_json, f, indent=2)
            
            # Clear selection history after successful save
            self.persistent_selections = {}
            self.selection_histories = {}
            
            # Update display to reflect cleared selections
            self.update_display()
            
            messagebox.showinfo("Success", f"Annotations saved to {save_path}\nSelection history has been cleared.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")
    
    def save_merged_boxes(self):
        """Save merged boxes to annotation file"""
        # Only available in merge mode
        if self.editor_mode != "merge":
            messagebox.showinfo("Mode Error", "Merged box saving is only available in merge mode")
            return
            
        if not self.merged_boxes:
            messagebox.showinfo("Info", "No merged boxes to save")
            return
        
        # Create merged annotation filename
        ann_dir = os.path.dirname(self.ann_file)
        ann_basename = os.path.basename(self.ann_file)
        merged_ann_file = os.path.join(ann_dir, f"merged_{ann_basename}")
        
        # Read original annotation lines
        original_lines = []
        with open(self.ann_file, 'r') as f:
            original_lines = f.readlines()
        
        # Create new annotation lines for merged boxes
        merged_lines = []
        for merged_box in self.merged_boxes:
            # Convert all values to integers
            frame_idx = int(merged_box['frame'])
            target_id = int(merged_box['id'])
            bbox_left = int(merged_box['bbox'][0])
            bbox_top = int(merged_box['bbox'][1])
            bbox_width = int(merged_box['bbox'][2])
            bbox_height = int(merged_box['bbox'][3])
            score = int(merged_box['score'])  # Convert score to integer
            category = int(merged_box['category'])
            truncation = int(merged_box['truncation'])
            occlusion = int(merged_box['occlusion'])
            
            # Format: <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
            line = f"{frame_idx},{target_id},{bbox_left},{bbox_top},{bbox_width},{bbox_height},{score},{category},{truncation},{occlusion}\n"
            merged_lines.append(line)
        
        # Modify original lines, only set source objects' score to 0 if option is enabled
        modified_original_lines = []
        for line in original_lines:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                track_id = int(parts[1])
                # Only modify source object's score when option is enabled
                if self.set_source_score_zero and track_id in self.merged_source_ids:
                    parts[6] = "0"  # Set score to 0
                    line = ','.join(parts) + '\n'
            modified_original_lines.append(line)
        
        # Use fixed save directory as initial directory
        initial_save_dir = self.save_default_dir
        if not initial_save_dir:
            # If no save directory is set, try to use annotation file directory
            initial_save_dir = os.path.dirname(self.ann_file)
        
        # Ask user for save location
        save_path = filedialog.asksaveasfilename(
            initialdir=initial_save_dir,
            initialfile=f"merged_{ann_basename}",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
        
        # Save selected directory as default directory for next time
        self.save_default_dir = os.path.dirname(save_path)
        
        # Save to file
        try:
            with open(save_path, 'w') as f:
                # Write modified original lines first
                for line in modified_original_lines:
                    f.write(line)
                
                # Write merged lines
                for line in merged_lines:
                    f.write(line)
            
            # Add information about source object scores in success message
            score_info = "Source objects' scores were set to 0." if self.set_source_score_zero else "Source objects' scores were preserved."
            messagebox.showinfo("Success", f"Merged boxes saved to {save_path}\n{score_info}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save merged boxes: {str(e)}")
    def on_close(self):
        """Handle window close event"""
        # Different save options depending on mode
        if self.editor_mode == "json":
            """ if messagebox.askyesno("Save before exit", "Do you want to save annotations before exiting?"):
                self.save_annotations() """
            has_selections = bool(self.persistent_selections) or any(self.selection_histories.values())
        
            if has_selections and messagebox.askyesno("Save before exit", "Do you want to save annotations before exiting?"):
                self.save_annotations()
        elif self.editor_mode == "merge":
            if self.merged_boxes and messagebox.askyesno("Save before exit", "Do you want to save merged boxes before exiting?"):
                self.save_merged_boxes()
        
        self.root.destroy()
        
    def run(self):
        """Run the application"""
        self.root.mainloop()            
def main():
    parser = argparse.ArgumentParser(description='Interactive MOT Annotation Editor')
    parser.add_argument('--img_dir', type=str, help='Directory containing images')
    parser.add_argument('--ann_file', type=str, help='Path to MOT format annotation file')
    parser.add_argument('--mode', type=str, choices=['json', 'merge'], help='Editor mode (json or merge)')
    args = parser.parse_args()
    
    editor = InteractiveMOTEditor()
    
    # If command line mode provided, set it
    if args.mode:
        editor.editor_mode = args.mode
    
    # If command line arguments provided, open the specified directory
    if args.img_dir:
        editor.img_dir = args.img_dir
        editor.ann_file = args.ann_file
        
        if editor.img_dir and not editor.ann_file:
            # Try to find annotation file automatically
            video_name = os.path.basename(editor.img_dir)
            parent_dir = os.path.dirname(os.path.dirname(editor.img_dir))
            possible_ann_file = os.path.join(parent_dir, "annotations", f"{video_name}.txt")
            
            if os.path.exists(possible_ann_file):
                editor.ann_file = possible_ann_file
                print(f"Found annotation file: {editor.ann_file}")
        
        if editor.img_dir and editor.ann_file:
            editor.load_dataset()
    
    editor.run()

if __name__ == "__main__":
    main()        