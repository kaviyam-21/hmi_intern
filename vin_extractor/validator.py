import re
from .config import ID_REGEX

class IDValidator:
    @staticmethod
    def clean_text(text: str) -> str:
        """Uppercase and remove all whitespace."""
        return re.sub(r'[\s\W_]+', '', text).upper()

    @staticmethod
    def validate(text: str) -> bool:
        """Check if the text matches the general ID format."""
        return bool(re.match(ID_REGEX, text))

class VINValidator(IDValidator):
    # Keep VIN-specific logic for backward compatibility if needed, 
    # but base it on general IDValidator for now.
    @staticmethod
    def process_and_validate(text: str) -> tuple[bool, str, bool]:
        """Backward compatibility stub."""
        cleaned = IDValidator.clean_text(text)
        is_valid = IDValidator.validate(cleaned)
        return is_valid, cleaned, False


    @staticmethod
    def is_check_digit_valid(vin: str) -> bool:
        """
        Validate VIN using the check digit at position 9 (ISO 3779 / North America).
        Returns True if check digit is valid, False otherwise.
        """
        if len(vin) != 17:
            return False
            
        # Transliteration map
        vals = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
            'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9, 'S': 2,
            'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9
        }
        
        # Weights
        weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
        
        try:
            total = 0
            for i in range(17):
                char = vin[i]
                if char.isdigit():
                    val = int(char)
                else:
                    val = vals.get(char, 0)
                total += val * weights[i]
            
            remainder = total % 11
            check_digit = 'X' if remainder == 10 else str(remainder)
            
            print(f"DEBUG Checksum: {vin} -> Remainder {remainder}, Expected {vin[8]}, Calculated {check_digit}", flush=True)
            return vin[8] == check_digit
        except (ValueError, KeyError):
            return False

    @staticmethod
    def try_fuzzy_correction(vin: str) -> str | None:
        """
        Try to fix common OCR misreads if they would make the checksum valid.
        This handles confusion pairs like (M, V), (B, 8, 3), (D, 0).
        """
        if len(vin) != 17:
            return None
            
        confusion_map = {
            'V': ['M'],
            'M': ['V'],
            '3': ['B'],
            '8': ['B'],
            '0': ['D', 'Q', 'O'], # O, Q are prohibited but might be read as 0
            'D': ['0'],
            'S': ['5'],
            '5': ['S'],
            'G': ['6'],
            '6': ['G'],
            'Z': ['2'],
            '2': ['Z']
        }
        
        # Identify indices with potential confusions
        risky_indices = [i for i, char in enumerate(vin) if char in confusion_map]
        risky_indices = risky_indices[:8]
        
        import itertools
        valid_candidates = []
        
        # Probabilistic weights for common misreads (lower cost = more likely)
        costs = {
            ('V', 'M'): 0.1, 
            ('3', 'B'): 0.2, 
            ('8', 'B'): 0.2,
            ('0', 'D'): 0.2,
            ('5', 'S'): 1.5, # Very high cost to change a digit to a letter
            ('S', '5'): 0.4,
            ('0', 'O'): 0.1,
            ('0', 'Q'): 0.1,
        }
        
        # Generate all combinations of alternatives
        for combo in itertools.product(*[ [vin[i]] + confusion_map[vin[i]] for i in risky_indices ]):
            candidate = list(vin)
            total_cost = 0.0
            for idx, new_char in zip(risky_indices, combo):
                orig_char = vin[idx]
                if new_char != orig_char:
                    total_cost += costs.get((orig_char, new_char), 1.0)
                candidate[idx] = new_char
            
            candidate_str = "".join(candidate)
            if VINValidator.validate(candidate_str) and VINValidator.is_check_digit_valid(candidate_str):
                # Heuristic: Bonus for common WMIs (like Ford 1FA, 2FM)
                if candidate_str.startswith(('1FA', '2FM', '3FA', 'WF0', 'JA1')):
                    total_cost -= 1.0 # Stronger bonus
                valid_candidates.append((candidate_str, total_cost))
                
        if not valid_candidates:
            return None
            
        # Return the one with the lowest total cost
        valid_candidates.sort(key=lambda x: x[1])
        return valid_candidates[0][0]

    @staticmethod
    def validate(text: str) -> bool:
        """Check if the text matches standard 17-character VIN format (ISO 3779)."""
        # ISO 3779: 17 chars, alphanumeric, NO I, O, Q
        return bool(re.match(r"^[A-HJ-NPR-Z0-9]{17}$", text))

    @staticmethod
    def process_and_validate(text: str) -> tuple[bool, str, bool]:
        """
        Full cleaning and validation pipeline.
        Returns: (is_format_valid, final_vin, is_checksum_valid)
        """
        # 1. Basic Cleaning
        cleaned = VINValidator.clean_text(text)
        
        candidates = []
        
        # Helper to check a string and its corrected version
        def evaluate(s):
            if len(s) == 17:
                is_f = VINValidator.validate(s)
                if not is_f:
                    return False, s, False
                
                # Check directly
                is_c = VINValidator.is_check_digit_valid(s)
                if is_c:
                    return True, s, True
                
                # Try fuzzy recovery if checksum fails
                fuzzy_fixed = VINValidator.try_fuzzy_correction(s)
                if fuzzy_fixed:
                    return True, fuzzy_fixed, True
                    
                return True, s, False
            return False, s, False

        # 2. Try whole string and corrected versions
        for s in [cleaned, VINValidator.correct_characters(cleaned)]:
            is_f, val, is_c = evaluate(s)
            if is_f: candidates.append((is_f, val, is_c))
            
            # 3. Robust Search for 17-char sub-windows
            if len(s) > 17:
                for i in range(len(s) - 16):
                    window = s[i:i+17]
                    is_f, val, is_c = evaluate(window)
                    if is_f: candidates.append((is_f, val, is_c))

        if not candidates:
            return False, cleaned, False

        # Sort candidates to prefer checksum valid ones
        candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)
        return candidates[0]
