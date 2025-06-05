"""
Text Processing Module for ChatterBox TTS
Handles pronunciation dictionary, sound effects, and text preprocessing
"""
import re
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PronunciationEntry:
    """Represents a pronunciation dictionary entry"""
    word: str
    pronunciation: str
    phonetic: Optional[str] = None
    notes: Optional[str] = None

class TextProcessor:
    """Advanced text processing for TTS input"""
    
    def __init__(self):
        self.pronunciation_dict = self._load_default_pronunciations()
        self.sound_effects = self._load_sound_effects()
        self.custom_pronunciations = {}
        
    def _load_default_pronunciations(self) -> Dict[str, str]:
        """Load default pronunciation dictionary"""
        return {
            # Common mispronunciations
            "AI": "A I",
            "API": "A P I", 
            "URL": "U R L",
            "HTTP": "H T T P",
            "HTTPS": "H T T P S",
            "GPU": "G P U",
            "CPU": "C P U",
            "RAM": "R A M",
            "SSD": "S S D",
            "HDD": "H D D",
            "USB": "U S B",
            "WiFi": "Wi-Fi",
            "iOS": "i O S",
            "macOS": "mac O S",
            "GitHub": "Git Hub",
            "YouTube": "You Tube",
            "LinkedIn": "Linked In",
            
            # Technical terms
            "TTS": "T T S",
            "NLP": "N L P",
            "ML": "M L",
            "CNN": "C N N",
            "RNN": "R N N",
            "LSTM": "L S T M",
            "GAN": "G A N",
            "VAE": "V A E",
            
            # Numbers and measurements
            "1st": "first",
            "2nd": "second", 
            "3rd": "third",
            "4th": "fourth",
            "5th": "fifth",
            "10th": "tenth",
            "100th": "hundredth",
            "1000th": "thousandth",
            
            # Common abbreviations
            "etc": "et cetera",
            "vs": "versus",
            "e.g.": "for example",
            "i.e.": "that is",
            "Mr.": "Mister",
            "Mrs.": "Missus",
            "Dr.": "Doctor",
            "Prof.": "Professor"
        }
    
    def _load_sound_effects(self) -> Dict[str, str]:
        """Load sound effect mappings"""
        return {
            # Laughter and positive sounds
            "[laugh]": "*laughs*",
            "[giggle]": "*giggles*",
            "[chuckle]": "*chuckles*",
            "[smile]": "",  # Silent but affects tone
            
            # Breathing and pauses
            "[breath]": "*takes a breath*",
            "[sigh]": "*sighs*",
            "[gasp]": "*gasps*",
            "[pause]": "... ",
            "[long_pause]": "...... ",
            
            # Throat sounds
            "[cough]": "*coughs*",
            "[clear_throat]": "*clears throat*",
            "[ahem]": "*ahem*",
            "[sneeze]": "*sneezes*",
            
            # Emotional sounds
            "[hmm]": "hmm",
            "[uh]": "uh",
            "[um]": "um",
            "[oh]": "oh",
            "[ah]": "ah",
            "[wow]": "wow",
            "[whoa]": "whoa",
            
            # Thinking sounds
            "[thinking]": "*thinking*",
            "[pondering]": "*pondering*",
            
            # Agreement/disagreement
            "[yes]": "yes",
            "[no]": "no",
            "[maybe]": "maybe",
            
            # Surprise/excitement
            "[gasp_excited]": "*excited gasp*",
            "[whistle]": "*whistles*",
            
            # Sadness
            "[sob]": "*sobs*",
            "[cry]": "*cries*",
            "[whimper]": "*whimpers*"
        }
    
    def add_pronunciation(self, word: str, pronunciation: str, phonetic: str = None, notes: str = None) -> bool:
        """
        Add a custom pronunciation
        
        Args:
            word: The word to add pronunciation for
            pronunciation: How the word should be pronounced
            phonetic: Optional phonetic representation
            notes: Optional notes about the pronunciation
            
        Returns:
            True if added successfully
        """
        try:
            entry = PronunciationEntry(
                word=word.lower(),
                pronunciation=pronunciation,
                phonetic=phonetic,
                notes=notes
            )
            self.custom_pronunciations[word.lower()] = pronunciation
            return True
        except Exception:
            return False
    
    def remove_pronunciation(self, word: str) -> bool:
        """Remove a custom pronunciation"""
        word_lower = word.lower()
        if word_lower in self.custom_pronunciations:
            del self.custom_pronunciations[word_lower]
            return True
        return False
    
    def get_pronunciation(self, word: str) -> Optional[str]:
        """Get pronunciation for a word"""
        word_lower = word.lower()
        
        # Check custom pronunciations first
        if word_lower in self.custom_pronunciations:
            return self.custom_pronunciations[word_lower]
        
        # Check default pronunciations
        if word_lower in self.pronunciation_dict:
            return self.pronunciation_dict[word_lower]
        
        return None
    
    def process_pronunciations(self, text: str) -> str:
        """Apply pronunciation dictionary to text"""
        processed_text = text
        
        # Apply custom pronunciations first
        for word, pronunciation in self.custom_pronunciations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(word) + r'\b'
            processed_text = re.sub(pattern, pronunciation, processed_text, flags=re.IGNORECASE)
        
        # Apply default pronunciations
        for word, pronunciation in self.pronunciation_dict.items():
            pattern = r'\b' + re.escape(word) + r'\b'
            processed_text = re.sub(pattern, pronunciation, processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def process_sound_effects(self, text: str) -> str:
        """Process sound effect tags in text"""
        processed_text = text
        
        for effect, replacement in self.sound_effects.items():
            # Case-insensitive replacement
            pattern = re.escape(effect)
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def preprocess_numbers(self, text: str) -> str:
        """Convert numbers to words for better TTS"""
        # Simple number conversion (can be expanded)
        number_map = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
            '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
        }
        
        processed_text = text
        
        # Convert standalone numbers
        for num, word in number_map.items():
            pattern = r'\b' + re.escape(num) + r'\b'
            processed_text = re.sub(pattern, word, processed_text)
        
        # Convert years (e.g., 2024 -> twenty twenty-four)
        year_pattern = r'\b(19|20)(\d{2})\b'
        def replace_year(match):
            century = match.group(1)
            year_part = match.group(2)
            
            if century == '19':
                century_word = 'nineteen'
            else:
                century_word = 'twenty'
            
            if year_part in number_map:
                year_word = number_map[year_part]
            else:
                # Handle 21-99
                tens = int(year_part[0])
                ones = int(year_part[1])
                tens_map = {2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty',
                           6: 'sixty', 7: 'seventy', 8: 'eighty', 9: 'ninety'}
                if tens in tens_map:
                    if ones == 0:
                        year_word = tens_map[tens]
                    else:
                        year_word = f"{tens_map[tens]}-{number_map[str(ones)]}"
                else:
                    year_word = year_part
            
            return f"{century_word} {year_word}"
        
        processed_text = re.sub(year_pattern, replace_year, processed_text)
        
        return processed_text
    
    def preprocess_urls_emails(self, text: str) -> str:
        """Convert URLs and emails to speakable format"""
        # URLs
        url_pattern = r'https?://[^\s]+'
        def replace_url(match):
            url = match.group(0)
            # Simplify URL for speech
            if 'github.com' in url:
                return 'GitHub link'
            elif 'youtube.com' in url or 'youtu.be' in url:
                return 'YouTube link'
            elif 'google.com' in url:
                return 'Google link'
            else:
                return 'website link'
        
        processed_text = re.sub(url_pattern, replace_url, text)
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        processed_text = re.sub(email_pattern, 'email address', processed_text)
        
        return processed_text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for TTS"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove or replace problematic characters
        text = text.replace('&', ' and ')
        text = text.replace('@', ' at ')
        text = text.replace('#', ' hashtag ')
        text = text.replace('$', ' dollars ')
        text = text.replace('%', ' percent ')
        
        # Fix common punctuation issues
        text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to ellipsis
        text = re.sub(r'!{2,}', '!', text)     # Multiple exclamations
        text = re.sub(r'\?{2,}', '?', text)    # Multiple questions
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def process_text(self, text: str, enable_pronunciations: bool = True, 
                    enable_sound_effects: bool = True, enable_preprocessing: bool = True) -> str:
        """
        Apply comprehensive text processing
        
        Args:
            text: Input text
            enable_pronunciations: Whether to apply pronunciation dictionary
            enable_sound_effects: Whether to process sound effects
            enable_preprocessing: Whether to apply general preprocessing
            
        Returns:
            Processed text ready for TTS
        """
        processed_text = text
        
        if enable_preprocessing:
            processed_text = self.clean_text(processed_text)
            processed_text = self.preprocess_numbers(processed_text)
            processed_text = self.preprocess_urls_emails(processed_text)
        
        if enable_sound_effects:
            processed_text = self.process_sound_effects(processed_text)
        
        if enable_pronunciations:
            processed_text = self.process_pronunciations(processed_text)
        
        return processed_text
    
    def get_processing_preview(self, text: str) -> Dict[str, str]:
        """Get a preview of how text will be processed"""
        return {
            "original": text,
            "cleaned": self.clean_text(text),
            "numbers_processed": self.preprocess_numbers(self.clean_text(text)),
            "urls_processed": self.preprocess_urls_emails(self.preprocess_numbers(self.clean_text(text))),
            "sound_effects": self.process_sound_effects(self.preprocess_urls_emails(self.preprocess_numbers(self.clean_text(text)))),
            "final": self.process_text(text)
        }
    
    def get_available_sound_effects(self) -> List[str]:
        """Get list of available sound effects"""
        return list(self.sound_effects.keys())
    
    def get_pronunciation_entries(self) -> Dict[str, str]:
        """Get all pronunciation entries"""
        all_pronunciations = self.pronunciation_dict.copy()
        all_pronunciations.update(self.custom_pronunciations)
        return all_pronunciations
    
    def export_custom_pronunciations(self) -> str:
        """Export custom pronunciations as JSON"""
        return json.dumps(self.custom_pronunciations, indent=2)
    
    def import_custom_pronunciations(self, json_data: str) -> Tuple[bool, str]:
        """Import custom pronunciations from JSON"""
        try:
            pronunciations = json.loads(json_data)
            if isinstance(pronunciations, dict):
                self.custom_pronunciations.update(pronunciations)
                return True, f"✅ Imported {len(pronunciations)} pronunciations"
            else:
                return False, "❌ Invalid JSON format"
        except Exception as e:
            return False, f"❌ Import failed: {str(e)}"

# Global text processor instance
text_processor = TextProcessor()
