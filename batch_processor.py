"""
Batch Processing Module for ChatterBox TTS
Handles multiple text inputs, queue management, and bulk generation
"""
import os
import tempfile
import zipfile
import json
import csv
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue
import time

@dataclass
class BatchItem:
    """Represents a single item in the batch processing queue"""
    id: str
    text: str
    reference_audio: Optional[str] = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    seed: Optional[int] = None
    preset: Optional[str] = None
    output_filename: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.id:
            self.id = f"batch_{int(time.time() * 1000)}"

class BatchProcessor:
    """Handles batch processing of TTS generation requests"""
    
    def __init__(self, tts_interface):
        self.tts_interface = tts_interface
        self.batch_queue = queue.Queue()
        self.completed_items = []
        self.failed_items = []
        self.is_processing = False
        self.current_item = None
        self.progress_callback = None
        self.total_items = 0
        self.processed_items = 0
        
    def add_text_batch(self, texts: List[str], **kwargs) -> List[str]:
        """
        Add multiple texts to the batch queue
        
        Args:
            texts: List of text strings to process
            **kwargs: Common parameters for all items
            
        Returns:
            List of batch item IDs
        """
        item_ids = []
        
        for i, text in enumerate(texts):
            if text.strip():  # Only add non-empty texts
                item = BatchItem(
                    id=f"batch_{int(time.time() * 1000)}_{i}",
                    text=text.strip(),
                    reference_audio=kwargs.get('reference_audio'),
                    exaggeration=kwargs.get('exaggeration', 0.5),
                    cfg_weight=kwargs.get('cfg_weight', 0.5),
                    seed=kwargs.get('seed'),
                    preset=kwargs.get('preset'),
                    output_filename=kwargs.get('output_filename_template', f"output_{i+1}")
                )
                
                self.batch_queue.put(item)
                item_ids.append(item.id)
        
        self.total_items += len(item_ids)
        return item_ids
    
    def add_csv_batch(self, csv_content: str, **kwargs) -> Tuple[List[str], str]:
        """
        Add batch items from CSV content
        
        CSV format: text,reference_audio,exaggeration,cfg_weight,seed,preset,filename
        
        Args:
            csv_content: CSV content as string
            **kwargs: Default parameters
            
        Returns:
            Tuple of (item_ids, status_message)
        """
        try:
            # Parse CSV
            csv_reader = csv.DictReader(csv_content.splitlines())
            item_ids = []
            
            for i, row in enumerate(csv_reader):
                text = row.get('text', '').strip()
                if not text:
                    continue
                
                item = BatchItem(
                    id=f"csv_batch_{int(time.time() * 1000)}_{i}",
                    text=text,
                    reference_audio=row.get('reference_audio') or kwargs.get('reference_audio'),
                    exaggeration=float(row.get('exaggeration', kwargs.get('exaggeration', 0.5))),
                    cfg_weight=float(row.get('cfg_weight', kwargs.get('cfg_weight', 0.5))),
                    seed=int(row.get('seed')) if row.get('seed') else kwargs.get('seed'),
                    preset=row.get('preset') or kwargs.get('preset'),
                    output_filename=row.get('filename', f"csv_output_{i+1}")
                )
                
                self.batch_queue.put(item)
                item_ids.append(item.id)
            
            self.total_items += len(item_ids)
            return item_ids, f"✅ Added {len(item_ids)} items from CSV"
            
        except Exception as e:
            return [], f"❌ Failed to parse CSV: {str(e)}"
    
    def add_json_batch(self, json_content: str) -> Tuple[List[str], str]:
        """
        Add batch items from JSON content
        
        Args:
            json_content: JSON content as string
            
        Returns:
            Tuple of (item_ids, status_message)
        """
        try:
            data = json.loads(json_content)
            item_ids = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and 'items' in data:
                items = data['items']
            else:
                return [], "❌ Invalid JSON structure. Expected list or object with 'items' key."
            
            for i, item_data in enumerate(items):
                if isinstance(item_data, str):
                    # Simple text list
                    text = item_data.strip()
                elif isinstance(item_data, dict):
                    # Full item object
                    text = item_data.get('text', '').strip()
                else:
                    continue
                
                if not text:
                    continue
                
                item = BatchItem(
                    id=f"json_batch_{int(time.time() * 1000)}_{i}",
                    text=text,
                    reference_audio=item_data.get('reference_audio') if isinstance(item_data, dict) else None,
                    exaggeration=item_data.get('exaggeration', 0.5) if isinstance(item_data, dict) else 0.5,
                    cfg_weight=item_data.get('cfg_weight', 0.5) if isinstance(item_data, dict) else 0.5,
                    seed=item_data.get('seed') if isinstance(item_data, dict) else None,
                    preset=item_data.get('preset') if isinstance(item_data, dict) else None,
                    output_filename=item_data.get('filename', f"json_output_{i+1}") if isinstance(item_data, dict) else f"json_output_{i+1}"
                )
                
                self.batch_queue.put(item)
                item_ids.append(item.id)
            
            self.total_items += len(item_ids)
            return item_ids, f"✅ Added {len(item_ids)} items from JSON"
            
        except Exception as e:
            return [], f"❌ Failed to parse JSON: {str(e)}"
    
    def process_batch(self, progress_callback: Optional[Callable] = None) -> None:
        """
        Process all items in the batch queue
        
        Args:
            progress_callback: Function to call with progress updates
        """
        if self.is_processing:
            return
        
        self.is_processing = True
        self.progress_callback = progress_callback
        self.processed_items = 0
        
        # Start processing in a separate thread
        processing_thread = threading.Thread(target=self._process_batch_worker)
        processing_thread.daemon = True
        processing_thread.start()
    
    def _process_batch_worker(self):
        """Worker function for batch processing"""
        try:
            while not self.batch_queue.empty():
                item = self.batch_queue.get()
                self.current_item = item
                
                # Update status
                item.status = "processing"
                self._update_progress()
                
                # Process the item
                try:
                    output_path, status_msg = self.tts_interface.generate_speech(
                        text=item.text,
                        audio_prompt_path=item.reference_audio,
                        exaggeration=item.exaggeration,
                        cfg_weight=item.cfg_weight,
                        seed=item.seed
                    )
                    
                    if output_path:
                        item.output_path = output_path
                        item.status = "completed"
                        item.completed_at = datetime.now().isoformat()
                        self.completed_items.append(item)
                    else:
                        item.status = "failed"
                        item.error_message = status_msg
                        self.failed_items.append(item)
                        
                except Exception as e:
                    item.status = "failed"
                    item.error_message = str(e)
                    self.failed_items.append(item)
                
                self.processed_items += 1
                self._update_progress()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
        finally:
            self.is_processing = False
            self.current_item = None
            self._update_progress()
    
    def _update_progress(self):
        """Update progress and call callback if provided"""
        if self.progress_callback:
            progress_info = {
                "total": self.total_items,
                "processed": self.processed_items,
                "completed": len(self.completed_items),
                "failed": len(self.failed_items),
                "current_item": self.current_item.text[:50] + "..." if self.current_item and len(self.current_item.text) > 50 else self.current_item.text if self.current_item else None,
                "is_processing": self.is_processing
            }
            self.progress_callback(progress_info)
    
    def get_batch_status(self) -> Dict[str, Any]:
        """Get current batch processing status"""
        return {
            "is_processing": self.is_processing,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "completed_items": len(self.completed_items),
            "failed_items": len(self.failed_items),
            "queue_size": self.batch_queue.qsize(),
            "current_item": self.current_item.text[:50] + "..." if self.current_item and len(self.current_item.text) > 50 else self.current_item.text if self.current_item else None
        }
    
    def create_batch_download(self) -> Tuple[Optional[str], str]:
        """
        Create a ZIP file with all completed batch items
        
        Returns:
            Tuple of (zip_path, status_message)
        """
        if not self.completed_items:
            return None, "❌ No completed items to download"
        
        try:
            # Create temporary ZIP file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                zip_path = tmp_file.name
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add audio files
                for item in self.completed_items:
                    if item.output_path and os.path.exists(item.output_path):
                        # Use custom filename if provided, otherwise use item ID
                        filename = f"{item.output_filename or item.id}.wav"
                        zip_file.write(item.output_path, filename)
                
                # Add batch report
                report = self._generate_batch_report()
                zip_file.writestr("batch_report.json", json.dumps(report, indent=2))
                zip_file.writestr("batch_summary.txt", self._generate_text_summary())
            
            return zip_path, f"✅ Created batch download with {len(self.completed_items)} files"
            
        except Exception as e:
            return None, f"❌ Failed to create batch download: {str(e)}"
    
    def _generate_batch_report(self) -> Dict[str, Any]:
        """Generate detailed batch processing report"""
        return {
            "batch_summary": {
                "total_items": self.total_items,
                "completed": len(self.completed_items),
                "failed": len(self.failed_items),
                "success_rate": len(self.completed_items) / self.total_items * 100 if self.total_items > 0 else 0
            },
            "completed_items": [asdict(item) for item in self.completed_items],
            "failed_items": [asdict(item) for item in self.failed_items],
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_text_summary(self) -> str:
        """Generate human-readable batch summary"""
        success_rate = len(self.completed_items) / self.total_items * 100 if self.total_items > 0 else 0
        
        summary = f"""
ChatterBox TTS Batch Processing Summary
======================================

📊 Overall Statistics:
- Total Items: {self.total_items}
- Completed: {len(self.completed_items)}
- Failed: {len(self.failed_items)}
- Success Rate: {success_rate:.1f}%

✅ Completed Items:
"""
        
        for item in self.completed_items:
            summary += f"- {item.output_filename or item.id}: \"{item.text[:50]}{'...' if len(item.text) > 50 else ''}\"\n"
        
        if self.failed_items:
            summary += f"\n❌ Failed Items:\n"
            for item in self.failed_items:
                summary += f"- {item.id}: \"{item.text[:50]}{'...' if len(item.text) > 50 else ''}\" - {item.error_message}\n"
        
        summary += f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return summary
    
    def clear_batch(self):
        """Clear all batch items and reset counters"""
        # Clear queue
        while not self.batch_queue.empty():
            try:
                self.batch_queue.get_nowait()
            except queue.Empty:
                break
        
        self.completed_items.clear()
        self.failed_items.clear()
        self.total_items = 0
        self.processed_items = 0
        self.current_item = None

# Global batch processor instance will be created when TTS interface is available
