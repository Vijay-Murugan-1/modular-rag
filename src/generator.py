import random
from src.models import Document

class SyntheticDatasetGenerator:
    """
    Generates a synthetic corpus of realistic, diverse, and duplicate-free 
    product documents optimized for evaluating RAG retrieval performance.
    """
    def __init__(self) -> None:
        self.catalog = {
            "Laptop": {
                "Gaming": {
                    "brands": ["ASUS", "MSI", "Lenovo", "Alienware"],
                    "features": [
                        "AI workload optimization", "Advanced thermal cooling",
                        "High refresh-rate display", "Premium build quality",
                        "Designed for AAA gaming", "High-speed NVMe storage"
                    ],
                    "attributes": {
                        "Processor": ["Intel Core Ultra 9", "AMD Ryzen 9"],
                        "Graphics": ["RTX 4070", "RTX 4080"],
                        "Memory": ["32GB", "64GB"],
                        "Storage": ["1TB SSD", "2TB SSD"],
                        "Battery": ["80Wh", "90Wh"]
                    }
                },
                "Business": {
                    "brands": ["Dell", "HP", "Lenovo"],
                    "features": [
                        "Lightweight design", "Long battery life",
                        "Enterprise-grade security", "Office productivity optimization",
                        "Enterprise-ready", "Business reliability"
                    ],
                    "attributes": {
                        "Processor": ["Intel Core Ultra 7", "AMD Ryzen 7"],
                        "Graphics": ["Intel Arc", "Intel Iris Xe"],
                        "Memory": ["16GB", "32GB"],
                        "Storage": ["512GB SSD", "1TB SSD"],
                        "Battery": ["70Wh", "80Wh"]
                    }
                },
                "Budget": {
                    "brands": ["Acer", "ASUS", "HP"],
                    "features": [
                        "Affordable pricing", "Energy efficient",
                        "Compact design", "Daily computing essentials",
                        "Perfect for students"
                    ],
                    "attributes": {
                        "Processor": ["Intel Core i3", "AMD Ryzen 5"],
                        "Graphics": ["Intel UHD"],
                        "Memory": ["8GB", "16GB"],
                        "Storage": ["256GB SSD", "512GB SSD"],
                        "Battery": ["50Wh", "60Wh"]
                    }
                }
            },
            "Smartphone": {
                "Flagship": {
                    "brands": ["Samsung", "Apple", "Google"],
                    "features": [
                        "Professional mobile photography", "Ultra-smooth AMOLED display",
                        "Fast wireless charging", "5G connectivity",
                        "AI-powered photography", "Premium flagship experience"
                    ],
                    "attributes": {
                        "Chipset": ["Snapdragon 8 Elite", "Apple A18 Pro", "Tensor G5"],
                        "Display": ["6.7-inch AMOLED"],
                        "Storage": ["256GB", "512GB"],
                        "Battery": ["5000mAh"],
                        "Camera": ["50MP Triple Camera"]
                    }
                },
                "Budget": {
                    "brands": ["Redmi", "POCO", "Realme"],
                    "features": [
                        "Excellent value for money", "Long battery life",
                        "Smooth everyday performance", "Modern design",
                        "Fast charging support"
                    ],
                    "attributes": {
                        "Chipset": ["Snapdragon 6 Gen 1", "Helio G99"],
                        "Display": ["6.5-inch LCD"],
                        "Storage": ["128GB"],
                        "Battery": ["5000mAh"],
                        "Camera": ["50MP Dual Camera"]
                    }
                }
            },
            "Tablet": {
                "Premium": {
                    "brands": ["Apple", "Samsung"],
                    "features": [
                        "Large immersive display", "Perfect for creativity",
                        "Excellent multimedia experience", "Powerful multitasking"
                    ],
                    "attributes": {
                        "Chipset": ["Apple M4", "Snapdragon X Elite"],
                        "Display": ["12.9-inch OLED"],
                        "Storage": ["256GB", "512GB"],
                        "Battery": ["9000mAh"]
                    }
                }
            },
            "Monitor": {
                "Gaming": {
                    "brands": ["LG", "ASUS", "MSI"],
                    "features": [
                        "Ultra-smooth gameplay", "High refresh-rate display",
                        "Low response time", "Immersive viewing experience"
                    ],
                    "attributes": {
                        "Resolution": ["1440p", "4K"],
                        "Refresh Rate": ["165Hz", "240Hz"],
                        "Panel": ["IPS", "OLED"],
                        "Size": ["27-inch", "32-inch"]
                    }
                }
            },
            "Keyboard": {
                "Mechanical": {
                    "brands": ["Keychron", "Corsair", "Logitech"],
                    "features": [
                        "Tactile typing experience", "Customizable RGB lighting",
                        "Durable mechanical switches", "Comfortable for long sessions"
                    ],
                    "attributes": {
                        "Switch Type": ["Red", "Brown", "Blue"],
                        "Layout": ["TKL", "Full Size", "75%"],
                        "Connectivity": ["USB-C", "Bluetooth"],
                        "Backlight": ["RGB", "White"]
                    }
                }
            }
        }
        
        self.brand_constraints = {
            "Apple": {"Chipset": ["Apple A18 Pro"]},
            "Samsung": {"Chipset": ["Snapdragon 8 Elite"]},
            "Google": {"Chipset": ["Tensor G5"]},
            "Redmi": {"Chipset": ["Snapdragon 6 Gen 1", "Helio G99"]},
            "POCO": {"Chipset": ["Snapdragon 6 Gen 1", "Helio G99"]},
            "Realme": {"Chipset": ["Snapdragon 6 Gen 1", "Helio G99"]},
            "Dell": {"Processor": ["Intel Core Ultra 7"]},
            "HP": {"Processor": ["Intel Core Ultra 7", "AMD Ryzen 7"]},
            "Lenovo": {"Processor": ["Intel Core Ultra 9", "AMD Ryzen 9", "Intel Core Ultra 7", "AMD Ryzen 7"]},
            "ASUS": {"Processor": ["Intel Core Ultra 9", "AMD Ryzen 9", "AMD Ryzen 5"]},
            "MSI": {"Processor": ["Intel Core Ultra 9", "AMD Ryzen 9"]},
            "Alienware": {"Processor": ["Intel Core Ultra 9", "AMD Ryzen 9"]},
            "Acer": {"Processor": ["Intel Core i3", "AMD Ryzen 5"]}
        }

        self.usage_profiles = {
            ("Laptop", "Gaming"): [
                "AAA gaming", "machine learning model training", "deep learning",
                "AI model development", "GPU-accelerated computing", "CUDA programming",
                "PyTorch and TensorFlow workloads", "data science"
            ],
            ("Laptop", "Business"): [
                "office productivity", "business workflows", "spreadsheet analysis",
                "document processing", "video conferencing", "remote collaboration"
            ],
            ("Laptop", "Budget"): [
                "students", "college assignments", "online learning", "web browsing",
                "document editing", "everyday computing"
            ],
            ("Smartphone", "Flagship"): [
                "mobile content creation", "social media", "4K video recording",
                "AI photography", "flagship mobile gaming", "professional videography"
            ],
            ("Smartphone", "Budget"): [
                "calling", "messaging", "social media", "video streaming", "web browsing"
            ]
        }
        self.used_signatures: set[tuple] = set()
        self.current_id: int = 1

    def _generate_features(self, category: str, profile: str, specifications: dict, feature_pool: list[str]) -> list[str]:
        selected = []
        if category == "Laptop":
            graphics = specifications.get("Graphics", "")
            processor = specifications.get("Processor", "")
            if "4080" in graphics or "4070" in graphics:
                selected.append("Designed for AAA gaming")
            if "Ultra 9" in processor or "Ryzen 9" in processor:
                selected.append("AI workload optimization")
            if profile == "Business":
                selected.append("Enterprise-ready")
            if profile == "Budget":
                selected.append("Affordable pricing")
        elif category == "Smartphone":
            camera = specifications.get("Camera", "")
            if "Triple" in camera:
                selected.append("Professional-grade camera system")

        remaining = [f for f in feature_pool if f not in selected]
        random.shuffle(remaining)
        while len(selected) < min(4, len(feature_pool)):
            selected.append(remaining.pop())
        return selected[:4]

    def _generate_description(self, name: str, category: str, profile: str, features: list[str], specifications: dict) -> str:
        usage = self.usage_profiles.get((category, profile), ["everyday tasks", "multimedia"])
        selected_usage = random.sample(usage, k=min(4, len(usage)))
        
        specs_str = ", ".join([f"{k} ({v})" for k, v in specifications.items()])
        return (
            f"The {name} is a high-performance {profile.lower()} category asset built with {specs_str}. "
            f"Engineered specifically for {', '.join(selected_usage[:-1])}, and {selected_usage[-1]}. "
            f"Highlights include: {', '.join(features)}."
        )

    def generate_document(self) -> Document:
        category = random.choice(list(self.catalog.keys()))
        profile = random.choice(list(self.catalog[category].keys()))
        pool = self.catalog[category][profile]
        brand = random.choice(pool["brands"])

        specifications = {}
        brand_rules = self.brand_constraints.get(brand, {})
        for attribute, values in pool["attributes"].items():
            if attribute in brand_rules:
                specifications[attribute] = random.choice(brand_rules[attribute])
            else:
                specifications[attribute] = random.choice(values)

        signature = (category, profile, brand, tuple(sorted(specifications.items())))
        if signature in self.used_signatures:
            return self.generate_document()
        self.used_signatures.add(signature)

        name = f"{brand} {profile} {category}"
        features = self._generate_features(category, profile, specifications, pool["features"])
        description = self._generate_description(name, category, profile, features, specifications)

        document = Document(
            document_id=self.current_id,
            name=name,
            category=category,
            features=features,
            specifications=specifications,
            description=description
        )
        self.current_id += 1
        return document

    def generate_dataset(self, num_documents: int) -> list[Document]:
        return [self.generate_document() for _ in range(num_documents)]