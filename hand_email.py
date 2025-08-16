import os
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque
from typing import List, Tuple, Optional
import imaplib
import email
from email.header import decode_header
import re

import cv2
import numpy as np
import mediapipe as mp


# ------------------------------
# Config and Acceleration
# ------------------------------

IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "mdbot2025@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "drvw hhkb hcmz vdhy")  # Prefer env var

# Prefer GPU/OpenCL acceleration when available
cv2.setUseOptimized(True)
try:
	cv2.ocl.setUseOpenCL(True)
except Exception:
	pass

CUDA_AVAILABLE = hasattr(cv2, "cuda") and getattr(cv2.cuda, "getCudaEnabledDeviceCount", lambda: 0)() > 0
if CUDA_AVAILABLE:
	print("[ACCEL] OpenCV CUDA is available and will be preferred for supported ops")
else:
	print("[ACCEL] OpenCV CUDA not available; using CPU/OpenCL where possible")


# ------------------------------
# IMAP Email Client (blocking API)
# ------------------------------

class EmailClient:
	def get_unread_emails(self, max_results: int = 25) -> List[dict]:
		try:
			mail = imaplib.IMAP4_SSL(IMAP_SERVER)
			mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
			mail.select("inbox")

			status, messages = mail.search(None, 'UNSEEN')
			if status != "OK" or not messages or not messages[0]:
				print("[IMAP] No unread emails found")
				mail.close()
				mail.logout()
				return []

			email_ids = messages[0].split()
			emails: List[dict] = []
			max_results = min(max_results, len(email_ids))
			print(f"[IMAP] Found {len(email_ids)} unread emails; fetching up to {max_results}")

			for i, eid in enumerate(reversed(email_ids)):
				if i >= max_results:
					break

				status, data = mail.fetch(eid, '(RFC822)')
				if status != "OK" or not data or not data[0]:
					print(f"[IMAP] Failed to fetch email {eid}")
					continue

				raw_email = data[0][1]
				try:
					msg = email.message_from_bytes(raw_email)
				except Exception as e:
					print(f"[IMAP] Error parsing email: {str(e)}")
					continue

				subject = self._decode_header(msg.get("Subject", ""))
				sender = self._decode_header(msg.get("From", ""))

				full_body = ""
				for part in msg.walk():
					if part.get_content_type() == "text/plain":
						try:
							body = part.get_payload(decode=True)
							if body:
								try:
									full_body = body.decode('utf-8', errors='replace')
								except UnicodeDecodeError:
									full_body = body.decode('latin-1', errors='replace')
								break
						except Exception as e:
							print(f"[IMAP] Error decoding body: {str(e)}")
							continue

				snippet = self._create_snippet(full_body)
				emails.append({
					"id": eid.decode() if isinstance(eid, bytes) else str(eid),
					"subject": subject,
					"from": sender,
					"snippet": snippet,
					"full_body": full_body
				})

			mail.close()
			mail.logout()
			return emails

		except Exception as e:
			print(f"[IMAP] Error fetching emails: {str(e)}")
			return []
	
	def _decode_header(self, header: str) -> str:
		try:
			decoded, encoding = decode_header(header)[0]
			if isinstance(decoded, bytes):
				return decoded.decode(encoding or "utf-8", errors='replace')
			return str(decoded)
		except:
			return header
	
	def _create_snippet(self, body: str) -> str:
		if not body:
			return ""
		clean_body = re.sub(r'>.*?\n', '', body)
		clean_body = re.sub(r'--\s*\n.*', '', clean_body)
		clean_body = re.sub(r'\n+', ' ', clean_body)
		clean_body = re.sub(r'\s+', ' ', clean_body)
		words = clean_body.split()
		if len(words) > 20:
			return " ".join(words[:20]) + "..."
		return clean_body


# ------------------------------
# Gesture Tracking
# ------------------------------

class HandTracker:
	def __init__(self):
		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands(
			static_image_mode=False,
			max_num_hands=1,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5
		)
		self.drawer = mp.solutions.drawing_utils
		self.drawer_styles = mp.solutions.drawing_styles

	def find_hand_landmarks(self, frame_bgr: np.ndarray) -> Tuple[Optional[List[Tuple[int, int]]], np.ndarray]:
		# Mirror the frame for natural interaction (left becomes right, right becomes left)
		frame_mirrored = cv2.flip(frame_bgr, 1)
		frame_rgb = cv2.cvtColor(frame_mirrored, cv2.COLOR_BGR2RGB)
		results = self.hands.process(frame_rgb)

		annotated = frame_mirrored.copy()
		if results.multi_hand_landmarks:
			hand_landmarks = results.multi_hand_landmarks[0]
			self.drawer.draw_landmarks(
				annotated,
				hand_landmarks,
				self.mp_hands.HAND_CONNECTIONS,
				self.drawer_styles.get_default_hand_landmarks_style(),
				self.drawer_styles.get_default_hand_connections_style()
			)
			h, w = frame_mirrored.shape[:2]
			points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
			return points, annotated

		return None, annotated


def euclidean_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
	return math.hypot(a[0] - b[0], a[1] - b[1])


class GestureInterpreter:
	def __init__(self, frame_width: int):
		self.frame_width = frame_width
		self.mode: str = "idle"  # idle | zoom | drag
		self.pinch_threshold_px = max(30, int(0.03 * frame_width))
		self.zoom_baseline_distance: Optional[float] = None
		self.zoom_baseline_scale: float = 1.0
		self.drag_offset: Tuple[int, int] = (0, 0)
		self.drag_baseline_y: Optional[int] = None

		# Enhanced swipe detection parameters (2D)
		self.palm_history = deque(maxlen=15)
		self.last_swipe_time = 0.0
		self.swipe_cooldown_sec = 0.2
		self.swipe_speed_threshold_px_per_s = 400.0
		self.swipe_distance_threshold_px = 30
		self.swipe_velocity_weight = 0.7
		self.min_swipe_points = 3
		self.swipe_confidence_threshold = 0.6

		# Tap detection
		self.tap_threshold_px = 20
		self.tap_time_threshold = 0.3
		self.last_tap_time = 0.0
		self.tap_cooldown = 0.5
		self.tap_start_pos: Optional[Tuple[int, int]] = None
		self.tap_start_time: Optional[float] = None

	def get_palm_center(self, pts: List[Tuple[int, int]]) -> Tuple[int, int]:
		indices = [0, 5, 9, 13, 17]
		x = int(sum(pts[i][0] for i in indices) / len(indices))
		y = int(sum(pts[i][1] for i in indices) / len(indices))
		return x, y

	def update_and_get_swipe_2d(self, palm_center: Tuple[int, int]) -> Tuple[int, int]:
		# Returns (horiz_dir: -1 left, +1 right, 0 none) and (vert_dir: -1 up, +1 down, 0 none)
		now = time.time()
		self.palm_history.append((now, palm_center[0], palm_center[1]))

		if len(self.palm_history) >= self.min_swipe_points:
			recent_points = list(self.palm_history)[-self.min_swipe_points:]
			distances_x: List[float] = []
			distances_y: List[float] = []
			velocities_x: List[float] = []
			velocities_y: List[float] = []

			for i in range(1, len(recent_points)):
				t0, x0, y0 = recent_points[i-1]
				t1, x1, y1 = recent_points[i]
				dt = max(1e-6, t1 - t0)
				dx = x1 - x0
				dy = y1 - y0
				distances_x.append(dx)
				distances_y.append(dy)
				velocities_x.append(dx / dt)
				velocities_y.append(dy / dt)

			total_dx = sum(distances_x)
			total_dy = sum(distances_y)
			avg_vx = sum(velocities_x) / len(velocities_x) if velocities_x else 0.0
			avg_vy = sum(velocities_y) / len(velocities_y) if velocities_y else 0.0

			if (now - self.last_swipe_time) > self.swipe_cooldown_sec:
				dist_score_x = min(1.0, abs(total_dx) / (self.swipe_distance_threshold_px * 2))
				vel_score_x = min(1.0, abs(avg_vx) / (self.swipe_speed_threshold_px_per_s * 0.5))
				score_x = (self.swipe_velocity_weight * vel_score_x + (1 - self.swipe_velocity_weight) * dist_score_x)

				dist_score_y = min(1.0, abs(total_dy) / (self.swipe_distance_threshold_px * 2))
				vel_score_y = min(1.0, abs(avg_vy) / (self.swipe_speed_threshold_px_per_s * 0.5))
				score_y = (self.swipe_velocity_weight * vel_score_y + (1 - self.swipe_velocity_weight) * dist_score_y)

				hdir = 0
				vdir = 0
				if score_x > self.swipe_confidence_threshold:
					hdir = +1 if total_dx > 0 else -1
				if score_y > self.swipe_confidence_threshold:
					vdir = +1 if total_dy > 0 else -1

				if hdir != 0 or vdir != 0:
					self.last_swipe_time = now
					return hdir, vdir

		return 0, 0

	def detect_single_tap(self, index_tip: Tuple[int, int], current_time: float) -> bool:
		if current_time - self.last_tap_time < self.tap_cooldown:
			return False
		if self.tap_start_pos is None:
			self.tap_start_pos = index_tip
			self.tap_start_time = current_time
			return False
		distance = euclidean_distance(self.tap_start_pos, index_tip)
		if distance > self.tap_threshold_px:
			self.tap_start_pos = None
			self.tap_start_time = None
			return False
		if current_time - self.tap_start_time > self.tap_time_threshold:
			self.tap_start_pos = None
			self.tap_start_time = None
			self.last_tap_time = current_time
			return True
		return False

	def reset_tap_tracking(self) -> None:
		self.tap_start_pos = None
		self.tap_start_time = None

	def detect_modes(self, pts: List[Tuple[int, int]], current_scale: float) -> Tuple[str, float, Tuple[int, int], Tuple[int, int]]:
		thumb_tip = pts[4]
		index_tip = pts[8]
		middle_tip = pts[12]
		palm_center = self.get_palm_center(pts)

		index_pinch_dist = euclidean_distance(thumb_tip, index_tip)
		middle_pinch_dist = euclidean_distance(thumb_tip, middle_tip)

		is_index_pinch = index_pinch_dist < self.pinch_threshold_px
		is_middle_pinch = middle_pinch_dist < self.pinch_threshold_px

		updated_scale = current_scale
		updated_window_pos = (0, 0)
		swipe_dirs = (0, 0)

		if self.mode == "idle":
			if is_index_pinch:
				self.mode = "zoom"
				self.zoom_baseline_distance = max(10.0, index_pinch_dist)
				self.zoom_baseline_scale = current_scale
			elif is_middle_pinch:
				self.mode = "drag"
				self.drag_offset = (palm_center[0] - updated_window_pos[0], palm_center[1] - updated_window_pos[1])
			else:
				swipe_dirs = self.update_and_get_swipe_2d(palm_center)
		elif self.mode == "zoom":
			if is_index_pinch and self.zoom_baseline_distance:
				ratio = index_pinch_dist / self.zoom_baseline_distance
				updated_scale = float(np.clip(self.zoom_baseline_scale * ratio, 0.6, 2.5))
			else:
				self.mode = "idle"
				self.zoom_baseline_distance = None
		elif self.mode == "drag":
			if is_middle_pinch:
				# Use vertical drag to scroll (hold middle pinch and move up/down)
				if self.drag_baseline_y is None:
					self.drag_baseline_y = palm_center[1]
				v_delta = palm_center[1] - self.drag_baseline_y
				# Return vdir as sign of v_delta to enable hold-scroll outside swipe logic
				vdir = 1 if v_delta > 10 else -1 if v_delta < -10 else 0
				swipe_dirs = (0, vdir)
			else:
				self.mode = "idle"
				self.drag_baseline_y = None

		return self.mode, updated_scale, updated_window_pos, swipe_dirs


# ------------------------------
# Email Panel (Single Window Overlay)
# ------------------------------

class EmailPanel:
	def __init__(self, width: int = 500, height: int = 380):
		self.width = width
		self.height = height
		self.scale = 1.0
		self.panel_x = 0
		self.panel_y = 0
		self.emails: List[dict] = []
		self.current_index: int = 0
		self.email_open: bool = False
		self.full_email_content: Optional[dict] = None
		self.close_button_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)

		self.scroll_offset = 0
		self.items_per_page = 6
		self.item_height = 90
		self.list_start_y = 100
		self.list_end_y = self.height - 80

		self.left_padding = 40
		self.right_padding = 40
		self.top_padding = 40
		self.bottom_padding = 40

		self.bg_color = (25, 25, 25)
		self.text_color = (240, 240, 240)
		self.accent_color = (80, 180, 255)
		self.highlight_color = (50, 100, 200)
		self.close_button_color = (220, 70, 70)
		self.separator_color = (60, 60, 60)

		self.close_button_flash_time = 0
		self.close_button_flash_duration = 0.3

		self.status_message = "Pinch to load emails"
		self.status_time = 0

		self.swipe_status = ""
		self.swipe_status_time = 0

		self.is_loading = False
		self.loading_start_time = 0.0

		# Content view scrolling (for full email)
		self.content_lines: List[str] = []
		self.content_scroll_index = 0

	def set_position(self, camera_width: int, camera_height: int) -> None:
		margin = 20
		self.panel_x = camera_width - self.width - margin
		self.panel_y = margin

	def get_panel_rect(self) -> Tuple[int, int, int, int]:
		return (self.panel_x, self.panel_y, self.width, self.height)

	def is_point_in_panel(self, x: int, y: int) -> bool:
		return (self.panel_x <= x <= self.panel_x + self.width and self.panel_y <= y <= self.panel_y + self.height)

	def convert_to_panel_coords(self, camera_x: int, camera_y: int) -> Tuple[int, int]:
		return camera_x - self.panel_x, camera_y - self.panel_y

	def set_loading(self, loading: bool) -> None:
		self.is_loading = loading
		self.loading_start_time = time.time()
		if loading:
			self.status_message = "Loading emails..."
			self.status_time = time.time()

	def set_emails(self, emails: List[dict]) -> None:
		self.emails = emails or []
		self.current_index = 0
		self.scroll_offset = 0
		self.email_open = False
		self.full_email_content = None
		self.content_lines = []
		self.content_scroll_index = 0
		if self.emails:
			self.status_message = f"Loaded {len(self.emails)} unread emails"
		else:
			self.status_message = "No unread emails found"
		self.status_time = time.time()
		self.set_loading(False)

	def set_swipe_status(self, message: str) -> None:
		self.swipe_status = message
		self.swipe_status_time = time.time()

	def next_email(self) -> None:
		if not self.emails:
			return
		self.current_index = (self.current_index + 1) % len(self.emails)
		self._ensure_current_email_visible()

	def prev_email(self) -> None:
		if not self.emails:
			return
		self.current_index = (self.current_index - 1) % len(self.emails)
		self._ensure_current_email_visible()

	def scroll_list(self, delta_items: int) -> None:
		if not self.emails:
			return
		self.scroll_offset = int(np.clip(self.scroll_offset + delta_items, 0, max(0, len(self.emails) - self.items_per_page)))
		# Keep selection within visible window
		end_visible = self.scroll_offset + self.items_per_page - 1
		self.current_index = int(np.clip(self.current_index, self.scroll_offset, min(end_visible, len(self.emails) - 1)))

	def _ensure_current_email_visible(self) -> None:
		if not self.emails:
			return
		visible_start = self.scroll_offset
		visible_end = self.scroll_offset + self.items_per_page
		if self.current_index < visible_start:
			self.scroll_offset = self.current_index
		elif self.current_index >= visible_end:
			self.scroll_offset = self.current_index - self.items_per_page + 1
		self.scroll_offset = int(np.clip(self.scroll_offset, 0, max(0, len(self.emails) - self.items_per_page)))

	def open_full_email(self, email_obj: dict) -> None:
		self.full_email_content = email_obj
		self.email_open = True
		button_size = 60
		self.close_button_rect = (self.width - button_size - self.right_padding, self.top_padding, button_size, button_size)
		self._prepare_content_lines()
		self.content_scroll_index = 0

	def close_full_email(self) -> None:
		self.email_open = False
		self.full_email_content = None
		self.content_lines = []
		self.content_scroll_index = 0

	def _prepare_content_lines(self) -> None:
		self.content_lines = []
		if not self.full_email_content:
			return
		body = self.full_email_content.get('full_body', self.full_email_content.get('snippet', ''))
		if not body:
			return
		body = body.replace('\r\n', '\n').replace('\r', '\n')
		body = re.sub(r'\n{3,}', '\n\n', body)
		max_width = self.width - self.left_padding - self.right_padding
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.7 * self.scale
		thickness = 1
		for paragraph in body.split('\n'):
			words = paragraph.split()
			current_line = ""
			for word in words:
				test_line = f"{current_line} {word}".strip()
				(width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
				if width <= max_width or not current_line:
					current_line = test_line
				else:
					self.content_lines.append(current_line)
					current_line = word
			if current_line:
				self.content_lines.append(current_line)
			# Paragraph gap
			self.content_lines.append("")

	def is_close_button_hit(self, x: int, y: int) -> bool:
		if not self.email_open:
			return False
		bx, by, bw, bh = self.close_button_rect
		return bx <= x <= bx + bw and by <= y <= by + bh

	def handle_close_button_pinch(self, pinch_x: int, pinch_y: int) -> bool:
		if not self.email_open:
			return False
		if self.is_close_button_hit(pinch_x, pinch_y):
			self.close_button_color = (255, 100, 100)
			self.close_button_flash_time = time.time()
			return True
		return False

	def reset_close_button_color(self) -> None:
		self.close_button_color = (220, 70, 70)

	def update_close_button_state(self) -> None:
		if self.close_button_flash_time > 0:
			if time.time() - self.close_button_flash_time > self.close_button_flash_duration:
				self.reset_close_button_color()
				self.close_button_flash_time = 0

	def render(self) -> np.ndarray:
		canvas = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

		# Status line (top-left)
		if time.time() - self.status_time < 2.5 and self.status_message:
			cv2.putText(canvas, self.status_message, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.accent_color, 2, cv2.LINE_AA)

		if self.is_loading:
			self._render_loading(canvas)
			return canvas

		if not self.email_open:
			self._render_email_list(canvas)
		else:
			self._render_email_content(canvas)

		return canvas

	def _render_loading(self, canvas: np.ndarray) -> None:
		cv2.putText(canvas, "Connecting to mail server...", (self.left_padding, self.height // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
		# Spinner
		angle = ((time.time() - self.loading_start_time) * 360) % 360
		center = (self.width // 2, self.height // 2 + 20)
		cv2.ellipse(canvas, center, (30, 30), 0, angle, angle + 270, self.accent_color, 4)

	def _render_email_list(self, canvas: np.ndarray) -> None:
		header = f"Unread Emails ({len(self.emails)})" if self.emails else "No Unread Emails"
		cv2.rectangle(canvas, (0, 0), (self.width, 80), (45, 45, 45), thickness=-1)
		cv2.putText(canvas, header, (self.left_padding, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.accent_color, 2, cv2.LINE_AA)

		if not self.emails:
			cv2.putText(canvas, "Pinch (Thumb+Index) to refresh", (self.left_padding, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 1, cv2.LINE_AA)
			return

		visible_emails = self.emails[self.scroll_offset:self.scroll_offset + self.items_per_page]
		for i, email_obj in enumerate(visible_emails):
			actual_index = self.scroll_offset + i
			y_offset = self.list_start_y + i * self.item_height
			if actual_index == self.current_index:
				cv2.rectangle(canvas, (self.left_padding, y_offset - 5), (self.width - self.right_padding, y_offset + self.item_height - 5), self.highlight_color, thickness=-1)
				cv2.circle(canvas, (self.left_padding + 20, y_offset + 20), 6, (255, 255, 255), -1)
				cv2.rectangle(canvas, (self.left_padding, y_offset - 5), (self.width - self.right_padding, y_offset + self.item_height - 5), (255, 255, 255), 2)
			if i > 0:
				cv2.line(canvas, (self.left_padding + 20, y_offset - 5), (self.width - self.right_padding - 20, y_offset - 5), self.separator_color, 2)

			subject = email_obj.get('subject', '')
			if len(subject) > 60:
				subject = subject[:57] + "..."
			cv2.putText(canvas, subject, (self.left_padding + 50, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 1, cv2.LINE_AA)

			sender = email_obj.get('from', '')
			if len(sender) > 55:
				sender = sender[:52] + "..."
			cv2.putText(canvas, sender, (self.left_padding + 50, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

			snippet = email_obj.get('snippet', '')
			if snippet and len(snippet) > 70:
				snippet = snippet[:67] + "..."
			if snippet:
				cv2.putText(canvas, snippet, (self.left_padding + 50, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1, cv2.LINE_AA)

		# Scroll indicators
		if self.scroll_offset > 0:
			cv2.putText(canvas, "↑", (self.width - 60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.accent_color, 2, cv2.LINE_AA)
			cv2.putText(canvas, f"{self.scroll_offset + 1}", (self.width - 80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
		if self.scroll_offset + self.items_per_page < len(self.emails):
			cv2.putText(canvas, "↓", (self.width - 60, self.height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.accent_color, 2, cv2.LINE_AA)
			cv2.putText(canvas, f"{len(self.emails) - self.scroll_offset - self.items_per_page}", (self.width - 80, self.height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

		if self.emails:
			position_text = f"{self.current_index + 1} / {len(self.emails)}"
			cv2.putText(canvas, position_text, (self.left_padding, self.height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.accent_color, 1, cv2.LINE_AA)

		cv2.putText(canvas, "Pinch: Open  |  Swipe LR: Navigate  |  Swipe UD: Scroll List", (self.left_padding, self.height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

	def _render_email_content(self, canvas: np.ndarray) -> None:
		if not self.full_email_content:
			self.email_open = False
			return
		email_obj = self.full_email_content

		cv2.rectangle(canvas, (0, 0), (self.width, 100), (45, 45, 45), thickness=-1)
		cv2.putText(canvas, "Email Content", (self.left_padding, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.accent_color, 2, cv2.LINE_AA)

		bx, by, bw, bh = self.close_button_rect
		cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), self.close_button_color, -1)
		cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (255, 255, 255), 3)
		cv2.rectangle(canvas, (bx + 3, by + 3), (bx + bw - 3, by + bh - 3), (255, 255, 255), 2)
		cv2.putText(canvas, "X", (bx + 18, by + 42), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
		cv2.putText(canvas, "Pinch to close", (bx - 10, by + bh + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

		y = 130
		y = self._draw_wrapped(canvas, f"Subject: {email_obj.get('subject', '')}", (self.left_padding, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * self.scale, self.text_color, self.width - self.left_padding - self.right_padding, thickness=2)
		y += 25
		y = self._draw_wrapped(canvas, f"From: {email_obj.get('from', '')}", (self.left_padding, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.scale, (200, 200, 200), self.width - self.left_padding - self.right_padding, thickness=1)
		y += 20
		cv2.line(canvas, (self.left_padding, y), (self.width - self.right_padding, y), self.separator_color, 2)
		y += 20

		# Body area window and scrolling
		body_top = y
		body_bottom = self.height - 60
		max_lines_fit = max(1, (body_bottom - body_top) // int(18 * (0.7 * self.scale) + 6))
		if self.content_lines:
			self.content_scroll_index = int(np.clip(self.content_scroll_index, 0, max(0, len(self.content_lines) - max_lines_fit)))
			visible_lines = self.content_lines[self.content_scroll_index:self.content_scroll_index + max_lines_fit]
			for line in visible_lines:
				cv2.putText(canvas, line, (self.left_padding, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.scale, (210, 210, 210), 1, cv2.LINE_AA)
				y += int(18 * (0.7 * self.scale)) + 6

		# Simple scrollbar indicator
		if self.content_lines and len(self.content_lines) > max_lines_fit:
			scrollbar_x = self.width - self.right_padding // 2
			scrollbar_top = body_top
			scrollbar_bottom = body_bottom
			cv2.line(canvas, (scrollbar_x, scrollbar_top), (scrollbar_x, scrollbar_bottom), (100, 100, 100), 3)
			ratio = (self.content_scroll_index + max_lines_fit) / max(1, len(self.content_lines))
			thumb_y = int(scrollbar_top + ratio * (scrollbar_bottom - scrollbar_top))
			cv2.circle(canvas, (scrollbar_x, thumb_y), 6, self.accent_color, -1)

		cv2.putText(canvas, "Swipe UD: Scroll  |  Pinch: Zoom  |  Middle Pinch: Drag (hold)", (self.left_padding, self.height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

	def _draw_wrapped(self, canvas: np.ndarray, text: str, origin: Tuple[int, int], font_face: int, font_scale: float, color: Tuple[int, int, int], max_width: int, thickness: int = 1, line_gap: int = 6) -> int:
		words = text.split()
		lines: List[str] = []
		current_line = ""
		for word in words:
			test_line = f"{current_line} {word}".strip()
			(width, _), _ = cv2.getTextSize(test_line, font_face, font_scale, thickness)
			if width <= max_width or not current_line:
				current_line = test_line
			else:
				lines.append(current_line)
				current_line = word
		if current_line:
			lines.append(current_line)
		x, y = origin
		for line in lines:
			cv2.putText(canvas, line, (x, y), font_face, font_scale, color, thickness, cv2.LINE_AA)
			y += int(18 * font_scale) + line_gap
		return y


# ------------------------------
# Asynchronous Email Fetcher
# ------------------------------

class EmailFetcher:
	def __init__(self, client: EmailClient):
		self.client = client
		self.executor = ThreadPoolExecutor(max_workers=1)
		self.lock = threading.Lock()
		self.in_flight: Optional[Future] = None

	def fetch_async(self, max_results: int = 25) -> None:
		with self.lock:
			if self.in_flight and not self.in_flight.done():
				return
			self.in_flight = self.executor.submit(self.client.get_unread_emails, max_results)

	def poll_result(self) -> Optional[List[dict]]:
		with self.lock:
			if not self.in_flight:
				return None
			if self.in_flight.done():
				try:
					result = self.in_flight.result()
					self.in_flight = None
					return result
				except Exception as e:
					print(f"[Fetcher] Error: {e}")
					self.in_flight = None
					return []
			return None


# ------------------------------
# App Orchestration (Single Window)
# ------------------------------

def _open_camera() -> Optional[cv2.VideoCapture]:
	# Prefer V4L2 on Linux, fallback to default
	cap = None
	try:
		cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
		if not cap or not cap.isOpened():
			raise RuntimeError("V4L2 open failed")
	except Exception:
		cap = cv2.VideoCapture(0)
	if not cap or not cap.isOpened():
		return None
	# Try higher fps via MJPG
	try:
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		cap.set(cv2.CAP_PROP_FOURCC, fourcc)
	except Exception:
		pass
	return cap


def main() -> None:
	cap = _open_camera()
	if not cap or not cap.isOpened():
		print("Could not open webcam.")
		return

	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	ret, test_frame = cap.read()
	if not ret:
		print("Could not read from webcam.")
		return
	frame_h, frame_w = test_frame.shape[:2]

	hand_tracker = HandTracker()
	gestures = GestureInterpreter(frame_width=frame_w)

	panel_width = min(600, frame_w // 2)
	panel_height = min(500, frame_h - 40)
	panel = EmailPanel(width=panel_width, height=panel_height)
	panel.set_position(frame_w, frame_h)

	email_client = EmailClient()
	fetcher = EmailFetcher(email_client)
	emails_loaded = False

	MAIN_WIN = "Hand Gesture Email Reader"
	cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(MAIN_WIN, frame_w, frame_h)

	last_pinch_time = 0.0
	pinch_cooldown = 1.0

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		# Hand tracking (mirroring done inside HandTracker)
		points, annotated = hand_tracker.find_hand_landmarks(frame)

		mode_label = f"Mode: {gestures.mode.upper()}"
		cv2.putText(annotated, mode_label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)

		current_time = time.time()
		if points:
			palm_center = gestures.get_palm_center(points)
			cv2.circle(annotated, palm_center, 8, (255, 0, 255), -1)
			cv2.putText(annotated, f"Palm: {palm_center[0]}, {palm_center[1]}", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1, cv2.LINE_AA)

			mode, new_scale, _, swipe_dirs = gestures.detect_modes(points, panel.scale)
			panel.scale = new_scale

			for idx in [0, 4, 8, 12]:
				cv2.circle(annotated, points[idx], 6, (0, 255, 0), -1)
				label = 'T' if idx == 4 else 'I' if idx == 8 else 'M' if idx == 12 else 'W'
				cv2.putText(annotated, label, (points[idx][0] + 10, points[idx][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

			thumb_tip = points[4]
			index_tip = points[8]
			index_pinch_dist = euclidean_distance(thumb_tip, index_tip)
			is_index_pinch = index_pinch_dist < gestures.pinch_threshold_px
			is_single_tap = gestures.detect_single_tap(index_tip, current_time)

			if is_index_pinch and (current_time - last_pinch_time) > pinch_cooldown:
				last_pinch_time = current_time
				cv2.circle(annotated, thumb_tip, 12, (0, 255, 0), 2)
				cv2.circle(annotated, index_tip, 12, (0, 255, 0), 2)
				cv2.line(annotated, thumb_tip, index_tip, (0, 255, 0), 2)
				cv2.putText(annotated, f"Pinch: {int(index_pinch_dist)}px", (thumb_tip[0] - 30, thumb_tip[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

				if panel.is_point_in_panel(index_tip[0], index_tip[1]):
					if panel.email_open:
						px, py = panel.convert_to_panel_coords(index_tip[0], index_tip[1])
						if panel.handle_close_button_pinch(px, py):
							panel.close_full_email()
							print("Email closed via pinch on X")
						else:
							print("Zoom mode active in email view")
					elif emails_loaded and panel.emails:
						panel.open_full_email(panel.emails[panel.current_index])
						print(f"Opened email: {panel.emails[panel.current_index].get('subject', '')}")
					else:
						panel.set_loading(True)
						fetcher.fetch_async(max_results=25)
						print("Loading emails in background...")
				else:
					print("Pinch outside email panel")

			elif is_single_tap and panel.email_open and panel.is_point_in_panel(index_tip[0], index_tip[1]):
				panel.close_full_email()
				cv2.circle(annotated, index_tip, 15, (0, 255, 255), 3)
				cv2.putText(annotated, "TAP - Email Closed", (index_tip[0] - 60, index_tip[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

			elif not is_index_pinch:
				gestures.reset_tap_tracking()

			# Horizontal and vertical swipes
			hdir, vdir = swipe_dirs
			if (hdir != 0 or vdir != 0) and (current_time - last_pinch_time) > pinch_cooldown:
				if not panel.email_open:
					# In list view: LR to change selection; UD to scroll list
					if hdir > 0:
						panel.next_email()
						panel.set_swipe_status("→ Next Email")
					elif hdir < 0:
						panel.prev_email()
						panel.set_swipe_status("← Previous Email")
					if vdir != 0:
						panel.scroll_list(delta_items=+1 if vdir > 0 else -1)
						panel.set_swipe_status("↓ Scroll" if vdir > 0 else "↑ Scroll")
				else:
					# In content view: UD to scroll content
					if vdir != 0:
						panel.content_scroll_index += (1 if vdir > 0 else -1) * 3
						panel.set_swipe_status("↓ Scroll" if vdir > 0 else "↑ Scroll")

		# Poll background email fetch results without blocking UI
		result = fetcher.poll_result()
		if result is not None:
			panel.set_emails(result)
			emails_loaded = True
			print(f"Background loaded {len(result)} unread emails")

		# Instruction panel (overlay, left side)
		instruction_bg = np.zeros((200, 300, 3), dtype=np.uint8)
		instruction_bg[:] = (0, 0, 0)
		cv2.rectangle(instruction_bg, (0, 0), (299, 199), (255, 255, 255), 2)
		instructions = [
			"GESTURE GUIDE:",
			"",
			"👆 Pinch (T+I): Load/Open/Close",
			"👆 Single Tap: Close Email",
			"👈 Swipe Left/Right: Prev/Next",
			"⬆ Swipe Up/Down: Scroll",
			"✋ Middle Pinch: Hold and move Up/Down to Scroll",
			"",
			"Mirrored View"
		]
		y_offset = 25
		for i, instruction in enumerate(instructions):
			color = (255, 255, 255) if i == 0 else (200, 200, 200) if i == 1 else (150, 255, 150)
			font_scale = 0.6 if i == 0 else 0.5
			thickness = 2 if i == 0 else 1
			cv2.putText(instruction_bg, instruction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
			y_offset += 20 if i == 0 else 18
		if annotated.shape[1] > 320 and annotated.shape[0] > 220:
			annotated[20:220, 20:320] = instruction_bg

		# Render email panel overlay and blend into a single window
		panel_img = panel.render()
		px, py, pw, ph = panel.get_panel_rect()
		if 0 <= px and 0 <= py and px + pw <= annotated.shape[1] and py + ph <= annotated.shape[0]:
			frame_region = annotated[py:py+ph, px:px+pw]
			alpha = 0.7
			blended = cv2.addWeighted(panel_img, alpha, frame_region, 1 - alpha, 0)
			annotated[py:py+ph, px:px+pw] = blended
			cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (255, 255, 255), 2)
			cv2.putText(annotated, "Email Panel", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

		try:
			cv2.imshow(MAIN_WIN, annotated)
		except Exception as e:
			print(f"[UI] imshow error: {e}")
			break

		panel.update_close_button_state()

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
		elif key == ord('r'):
			panel.set_loading(True)
			fetcher.fetch_async(max_results=25)
			print("Manual refresh queued...")

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()