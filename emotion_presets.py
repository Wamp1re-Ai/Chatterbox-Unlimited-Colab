"""
Emotion and Style Presets for ChatterBox TTS
Provides pre-configured emotion profiles and speaking styles
"""
from typing import Dict, List, Tuple, Any
import json
import os

class EmotionPresets:
    """Manages emotion and style presets for TTS generation"""
    
    def __init__(self):
        self.presets = self._load_default_presets()
        self.custom_presets = {}
        
    def _load_default_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load default emotion and style presets"""
        return {
            # Emotion Presets
            "neutral": {
                "type": "emotion",
                "name": "Neutral",
                "description": "Calm, balanced speech with no particular emotion",
                "exaggeration": 0.3,
                "cfg_weight": 0.5,
                "icon": "😐",
                "tags": ["basic", "professional"]
            },
            "happy": {
                "type": "emotion", 
                "name": "Happy",
                "description": "Cheerful, upbeat, and positive tone",
                "exaggeration": 0.7,
                "cfg_weight": 0.6,
                "icon": "😊",
                "tags": ["positive", "energetic"]
            },
            "excited": {
                "type": "emotion",
                "name": "Excited", 
                "description": "High energy, enthusiastic delivery",
                "exaggeration": 0.9,
                "cfg_weight": 0.7,
                "icon": "🤩",
                "tags": ["energetic", "dynamic"]
            },
            "sad": {
                "type": "emotion",
                "name": "Sad",
                "description": "Melancholic, slower paced speech",
                "exaggeration": 0.6,
                "cfg_weight": 0.3,
                "icon": "😢",
                "tags": ["emotional", "slow"]
            },
            "angry": {
                "type": "emotion",
                "name": "Angry",
                "description": "Intense, forceful delivery",
                "exaggeration": 0.8,
                "cfg_weight": 0.8,
                "icon": "😠",
                "tags": ["intense", "forceful"]
            },
            "calm": {
                "type": "emotion",
                "name": "Calm",
                "description": "Peaceful, relaxed, soothing tone",
                "exaggeration": 0.2,
                "cfg_weight": 0.4,
                "icon": "😌",
                "tags": ["relaxed", "soothing"]
            },
            "surprised": {
                "type": "emotion",
                "name": "Surprised",
                "description": "Sudden, unexpected tone with emphasis",
                "exaggeration": 0.8,
                "cfg_weight": 0.6,
                "icon": "😲",
                "tags": ["dynamic", "emphasis"]
            },
            "confident": {
                "type": "emotion",
                "name": "Confident",
                "description": "Strong, assured, authoritative delivery",
                "exaggeration": 0.6,
                "cfg_weight": 0.7,
                "icon": "😎",
                "tags": ["strong", "professional"]
            },
            
            # Speaking Style Presets
            "conversational": {
                "type": "style",
                "name": "Conversational",
                "description": "Natural, casual conversation style",
                "exaggeration": 0.4,
                "cfg_weight": 0.5,
                "icon": "💬",
                "tags": ["natural", "casual"]
            },
            "news_anchor": {
                "type": "style",
                "name": "News Anchor",
                "description": "Professional, clear, authoritative news delivery",
                "exaggeration": 0.3,
                "cfg_weight": 0.6,
                "icon": "📺",
                "tags": ["professional", "clear"]
            },
            "storyteller": {
                "type": "style",
                "name": "Storyteller",
                "description": "Engaging, dramatic narrative style",
                "exaggeration": 0.7,
                "cfg_weight": 0.6,
                "icon": "📚",
                "tags": ["dramatic", "engaging"]
            },
            "meditation": {
                "type": "style",
                "name": "Meditation Guide",
                "description": "Slow, peaceful, mindful delivery",
                "exaggeration": 0.2,
                "cfg_weight": 0.3,
                "icon": "🧘",
                "tags": ["peaceful", "slow"]
            },
            "audiobook": {
                "type": "style",
                "name": "Audiobook Narrator",
                "description": "Clear, consistent, engaging reading style",
                "exaggeration": 0.4,
                "cfg_weight": 0.5,
                "icon": "🎧",
                "tags": ["clear", "consistent"]
            },
            "presentation": {
                "type": "style",
                "name": "Presentation",
                "description": "Professional, engaging presentation style",
                "exaggeration": 0.5,
                "cfg_weight": 0.6,
                "icon": "📊",
                "tags": ["professional", "engaging"]
            },
            "commercial": {
                "type": "style",
                "name": "Commercial",
                "description": "Persuasive, energetic advertising style",
                "exaggeration": 0.8,
                "cfg_weight": 0.7,
                "icon": "📢",
                "tags": ["persuasive", "energetic"]
            },
            "documentary": {
                "type": "style",
                "name": "Documentary",
                "description": "Informative, authoritative documentary style",
                "exaggeration": 0.4,
                "cfg_weight": 0.6,
                "icon": "🎬",
                "tags": ["informative", "authoritative"]
            }
        }
    
    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get a specific preset by name"""
        if preset_name in self.presets:
            return self.presets[preset_name]
        elif preset_name in self.custom_presets:
            return self.custom_presets[preset_name]
        else:
            return self.presets["neutral"]  # Default fallback
    
    def get_presets_by_type(self, preset_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all presets of a specific type (emotion or style)"""
        filtered_presets = {}
        
        # Check default presets
        for name, preset in self.presets.items():
            if preset.get("type") == preset_type:
                filtered_presets[name] = preset
        
        # Check custom presets
        for name, preset in self.custom_presets.items():
            if preset.get("type") == preset_type:
                filtered_presets[name] = preset
                
        return filtered_presets
    
    def get_emotion_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get all emotion presets"""
        return self.get_presets_by_type("emotion")
    
    def get_style_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get all style presets"""
        return self.get_presets_by_type("style")
    
    def get_all_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get all presets (default + custom)"""
        all_presets = self.presets.copy()
        all_presets.update(self.custom_presets)
        return all_presets
    
    def create_custom_preset(
        self,
        name: str,
        preset_type: str,
        description: str,
        exaggeration: float,
        cfg_weight: float,
        icon: str = "🎭",
        tags: List[str] = None
    ) -> bool:
        """Create a custom preset"""
        if tags is None:
            tags = ["custom"]
            
        custom_preset = {
            "type": preset_type,
            "name": name,
            "description": description,
            "exaggeration": max(0.0, min(1.0, exaggeration)),  # Clamp to 0-1
            "cfg_weight": max(0.0, min(1.0, cfg_weight)),      # Clamp to 0-1
            "icon": icon,
            "tags": tags + ["custom"]
        }
        
        self.custom_presets[name.lower().replace(" ", "_")] = custom_preset
        return True
    
    def delete_custom_preset(self, preset_name: str) -> bool:
        """Delete a custom preset"""
        if preset_name in self.custom_presets:
            del self.custom_presets[preset_name]
            return True
        return False
    
    def get_preset_parameters(self, preset_name: str) -> Tuple[float, float]:
        """Get exaggeration and cfg_weight for a preset"""
        preset = self.get_preset(preset_name)
        return preset.get("exaggeration", 0.5), preset.get("cfg_weight", 0.5)
    
    def get_preset_info(self, preset_name: str) -> str:
        """Get formatted information about a preset"""
        preset = self.get_preset(preset_name)
        
        info = f"""
        {preset.get('icon', '🎭')} **{preset.get('name', 'Unknown')}**
        
        📝 {preset.get('description', 'No description available')}
        
        ⚙️ **Parameters:**
        - Exaggeration: {preset.get('exaggeration', 0.5):.1f}
        - CFG Weight: {preset.get('cfg_weight', 0.5):.1f}
        
        🏷️ **Tags:** {', '.join(preset.get('tags', []))}
        """
        
        return info
    
    def search_presets(self, query: str) -> Dict[str, Dict[str, Any]]:
        """Search presets by name, description, or tags"""
        query = query.lower()
        results = {}
        
        all_presets = self.get_all_presets()
        
        for name, preset in all_presets.items():
            # Search in name
            if query in name.lower():
                results[name] = preset
                continue
                
            # Search in description
            if query in preset.get('description', '').lower():
                results[name] = preset
                continue
                
            # Search in tags
            tags = preset.get('tags', [])
            if any(query in tag.lower() for tag in tags):
                results[name] = preset
                continue
        
        return results
    
    def get_preset_suggestions(self, text: str) -> List[str]:
        """Get preset suggestions based on text content"""
        text_lower = text.lower()
        suggestions = []
        
        # Emotion keywords
        emotion_keywords = {
            "happy": ["happy", "joy", "excited", "great", "wonderful", "amazing"],
            "sad": ["sad", "sorry", "unfortunately", "tragic", "disappointed"],
            "angry": ["angry", "furious", "outraged", "terrible", "awful"],
            "excited": ["exciting", "incredible", "fantastic", "wow", "amazing"],
            "calm": ["peaceful", "relax", "calm", "gentle", "soothing"],
            "confident": ["confident", "strong", "powerful", "certain", "sure"]
        }
        
        # Style keywords
        style_keywords = {
            "news_anchor": ["news", "report", "breaking", "update", "announcement"],
            "storyteller": ["story", "once upon", "tale", "adventure", "journey"],
            "commercial": ["buy", "sale", "offer", "deal", "product", "service"],
            "presentation": ["presentation", "data", "analysis", "results", "findings"],
            "meditation": ["breathe", "relax", "mindful", "peace", "meditation"]
        }
        
        # Check for emotion keywords
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                suggestions.append(emotion)
        
        # Check for style keywords
        for style, keywords in style_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                suggestions.append(style)
        
        # Default suggestions if none found
        if not suggestions:
            suggestions = ["conversational", "neutral"]
        
        return suggestions[:3]  # Return top 3 suggestions

# Global emotion presets instance
emotion_presets = EmotionPresets()
