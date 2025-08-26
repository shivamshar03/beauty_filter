class DressManager:
    """Enhanced dress management with improved overlay from dresse filter.py"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.assets_dir = Path(config.ASSETS_DIR)

        self.seasons = ["spring", "summer", "fall", "winter"]
        self.seasonal_dresses = {}
        self.current_season_idx = 0
        self.current_dress_idx = 0

        self._load_dresses()

    def _load_dresses(self):
        """Load dresses using the improved logic from dresse filter.py"""
        try:
            # Load dresses with RGBA support (improved from dresse filter.py)
            exts = {".png", ".PNG"}

            # Create main directory if it doesn't exist
            self.assets_dir.mkdir(parents=True, exist_ok=True)

            files = sorted([p for p in self.assets_dir.glob("*") if p.suffix in exts])
            imgs = []

            for p in files:
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # BGRA
                if img is None:
                    self.logger.warning(f"Skipping {p.name}: could not load image")
                    continue

                # Handle different image formats
                if len(img.shape) == 3 and img.shape[2] == 4:
                    # Already BGRA
                    imgs.append((p.name, img))
                elif len(img.shape) == 3 and img.shape[2] == 3:
                    # Convert BGR to BGRA
                    bgra_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    imgs.append((p.name, bgra_img))
                    self.logger.info(f"Converted {p.name} from BGR to BGRA")
                else:
                    self.logger.warning(f"Skipping {p.name}: unsupported format")
                    continue

            # Distribute dresses across seasons
            self.seasonal_dresses = {season: [] for season in self.seasons}

            for name, img in imgs:
                # Distribute dresses across seasons using hash for consistency
                season_idx = hash(name) % len(self.seasons)
                season = self.seasons[season_idx]
                self.seasonal_dresses[season].append((name, img))

            total_loaded = len(imgs)
            self.logger.info(f"Total dresses loaded: {total_loaded}")

            for season in self.seasons:
                count = len(self.seasonal_dresses[season])
                self.logger.info(f"  {season}: {count} dresses")

            if total_loaded == 0:
                self.logger.warning(f"No valid PNG files found in {self.assets_dir}")
                self._create_sample_info()

        except Exception as e:
            self.logger.error(f"Error loading dresses: {e}")

    def _create_sample_info(self):
        """Show info about expected file structure"""
        print(f"\n[INFO] No dress files found in {self.assets_dir}")
        print("[INFO] Please add PNG files (with transparent backgrounds) to continue")
        print("[INFO] Supported formats: PNG with alpha channel")

    def get_current_dress(self) -> Optional[Tuple[str, np.ndarray]]:
        """Get current dress with error handling"""
        try:
            season = self.seasons[self.current_season_idx]
            dresses = self.seasonal_dresses.get(season, [])

            if dresses and 0 <= self.current_dress_idx < len(dresses):
                return dresses[self.current_dress_idx]
            return None

        except Exception as e:
            self.logger.error(f"Error getting current dress: {e}")
            return None

    def next_dress(self) -> Optional[str]:
        """Navigate to next dress safely"""
        try:
            season = self.seasons[self.current_season_idx]
            dresses = self.seasonal_dresses.get(season, [])

            if dresses:
                self.current_dress_idx = (self.current_dress_idx + 1) % len(dresses)
                return dresses[self.current_dress_idx][0]
            return None

        except Exception as e:
            self.logger.error(f"Error navigating to next dress: {e}")
            return None

    def previous_dress(self) -> Optional[str]:
        """Navigate to previous dress safely"""
        try:
            season = self.seasons[self.current_season_idx]
            dresses = self.seasonal_dresses.get(season, [])

            if dresses:
                self.current_dress_idx = (self.current_dress_idx - 1) % len(dresses)
                return dresses[self.current_dress_idx][0]
            return None

        except Exception as e:
            self.logger.error(f"Error navigating to previous dress: {e}")
            return None

    def next_season(self) -> str:
        """Navigate to next season safely"""
        try:
            self.current_season_idx = (self.current_season_idx + 1) % len(self.seasons)
            self.current_dress_idx = 0  # Reset to first dress in new season
            season = self.seasons[self.current_season_idx]
            return season

        except Exception as e:
            self.logger.error(f"Error navigating to next season: {e}")
            return self.seasons[0]

    def get_season_info(self) -> Tuple[str, int]:
        """Get current season and dress count"""
        try:
            season = self.seasons[self.current_season_idx]
            dress_count = len(self.seasonal_dresses.get(season, []))
            return season, dress_count
        except Exception as e:
            self.logger.error(f"Error getting season info: {e}")
            return "spring", 0

    def find_best_dress(self, recommended_colors: List[str]) -> Optional[Tuple[int, str, int]]:
        """Find best matching dress for recommended colors"""
        try:
            season = self.seasons[self.current_season_idx]
            dresses = self.seasonal_dresses.get(season, [])

            if not dresses or not recommended_colors:
                return None

            best_score = 0
            best_index = 0

            for i, (name, _) in enumerate(dresses):
                score = 0
                name_lower = name.lower()

                # Check for color matches in filename
                for color in recommended_colors:
                    if color.lower() in name_lower:
                        score += 10
                    else:
                        # Check for partial matches
                        for word in color.lower().split():
                            if len(word) > 2 and word in name_lower:
                                score += 5
                                break

                if score > best_score:
                    best_score = score
                    best_index = i

            if best_score > 0:
                return best_index, dresses[best_index][0], best_score
            return None

        except Exception as e:
            self.logger.error(f"Error finding best dress: {e}")
            return None
