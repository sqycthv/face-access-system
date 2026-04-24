import os
import cv2
import json
import time
import base64
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import flet as ft
import numpy as np


ICONS = getattr(ft, "Icons", getattr(ft, "icons", None))
NAV_BAR_DEST = getattr(ft, "NavigationBarDestination", getattr(ft, "NavigationDestination", None))

# 1x1 transparent PNG placeholder for ft.Image on Android
EMPTY_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def icon(name: str):
    if ICONS is None:
        return name
    value = getattr(ICONS, name, None)
    if value is not None:
        return value
    lower_name = name.lower()
    return getattr(ICONS, lower_name, lower_name)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "access_app_data_v3")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
LOGS_FILE = os.path.join(DATA_DIR, "logs.json")
REQUESTS_FILE = os.path.join(DATA_DIR, "requests.json")
TEMP_FILE = os.path.join(DATA_DIR, "temp_passes.json")
CAPTURE_DIR = os.path.join(DATA_DIR, "captures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)

BG = "#F4F7FB"
SURFACE = "#FFFFFF"
SURFACE_2 = "#F8FAFD"
TEXT = "#101828"
MUTED = "#667085"
BORDER = "#E4E7EC"
PRIMARY = "#3563E9"
PRIMARY_SOFT = "#EEF4FF"
GREEN = "#16A34A"
GREEN_SOFT = "#ECFDF3"
RED = "#EF4444"
RED_SOFT = "#FEF2F2"
AMBER = "#F59E0B"
AMBER_SOFT = "#FFF7ED"
BLUE = "#0EA5E9"
BLUE_SOFT = "#F0F9FF"
PURPLE = "#8B5CF6"
PURPLE_SOFT = "#F5F3FF"

ROLES = ["student", "teacher", "admin"]
ZONES = [
    "Главный вход",
    "Аудитория 302",
    "Аудитория 212",
    "Лаборатория 208",
    "Деканат",
    "Библиотека",
    "Серверная",
]
ROLE_DEFAULT_ZONES = {
    "student": ["Главный вход", "Аудитория 302",  "Аудитория 212", "Лаборатория 208", "Библиотека"],
    "teacher": ["Главный вход", "Аудитория 302", "Аудитория 212", "Лаборатория 208", "Библиотека", "Деканат"],
    "admin": ZONES[:],
}

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def now() -> datetime:
    return datetime.now()


def now_str() -> str:
    return now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_dt(value: Optional[str]) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return value


def json_load(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def json_save(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}"


def image_file_to_data_uri(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def save_frame(frame: np.ndarray, prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(CAPTURE_DIR, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def create_fallback_image(text: str = "NO CAMERA") -> str:
    canvas = np.ones((300, 420, 3), dtype=np.uint8) * 244
    cv2.rectangle(canvas, (20, 20), (400, 280), (53, 99, 233), 2)
    cv2.circle(canvas, (210, 120), 52, (210, 220, 240), -1)
    cv2.rectangle(canvas, (130, 195), (290, 245), (220, 228, 245), -1)
    cv2.putText(canvas, text[:22], (85, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 90, 120), 2)
    return save_frame(canvas, "fallback")


def capture_attempt_snapshot() -> Tuple[str, bool]:
    indexes = [0, 1, 2]
    for idx in indexes:
        cap = None
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        except Exception:
            try:
                cap = cv2.VideoCapture(idx)
            except Exception:
                cap = None
        if cap is None:
            continue
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    frame = cv2.flip(frame, 1)
                    path = save_frame(frame, "attempt")
                    cap.release()
                    return path, True
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
    return create_fallback_image(), False


def crop_biggest_face(frame: np.ndarray) -> Optional[np.ndarray]:
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(70, 70))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        pad = int(min(w, h) * 0.18)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        return frame[y1:y2, x1:x2]
    except Exception:
        return None


def make_signature(face_bgr: np.ndarray) -> List[float]:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    vec = small.astype(np.float32).flatten()
    mean = float(np.mean(vec))
    std = float(np.std(vec))
    if std < 1e-6:
        std = 1.0
    vec = (vec - mean) / std
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        norm = 1.0
    vec = vec / norm
    return vec.astype(float).tolist()


def cosine_similarity(sig1: List[float], sig2: List[float]) -> float:
    if not sig1 or not sig2:
        return 0.0
    a = np.array(sig1, dtype=np.float32)
    b = np.array(sig2, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def classify_time_of_day(dt_str: str) -> str:
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return "day"
    h = dt.hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 18:
        return "day"
    return "evening"


class DataStore:
    def __init__(self) -> None:
        self.users: List[Dict[str, Any]] = json_load(USERS_FILE, [])
        self.logs: List[Dict[str, Any]] = json_load(LOGS_FILE, [])
        self.requests: List[Dict[str, Any]] = json_load(REQUESTS_FILE, [])
        self.temp_passes: List[Dict[str, Any]] = json_load(TEMP_FILE, [])
        self.ensure_defaults()
        self.cleanup_expired_passes()

    def save_all(self) -> None:
        json_save(USERS_FILE, self.users)
        json_save(LOGS_FILE, self.logs)
        json_save(REQUESTS_FILE, self.requests)
        json_save(TEMP_FILE, self.temp_passes)

    def ensure_defaults(self) -> None:
        if self.users:
            return
        student_pass_end = (now() + timedelta(minutes=8)).isoformat(timespec="minutes")
        self.users = [
            {
                "login": "student",
                "password": "1234",
                "name": "Student",
                "role": "student",
                "zones": ROLE_DEFAULT_ZONES["student"],
                "signature": [],
                "photo_path": "",
                "created_at": now_str(),
            },
            {
                "login": "teacher",
                "password": "1234",
                "name": "Teacher",
                "role": "teacher",
                "zones": ROLE_DEFAULT_ZONES["teacher"],
                "signature": [],
                "photo_path": "",
                "created_at": now_str(),
            },
            {
                "login": "admin",
                "password": "1234",
                "name": "Admin",
                "role": "admin",
                "zones": ROLE_DEFAULT_ZONES["admin"],
                "signature": [],
                "photo_path": "",
                "created_at": now_str(),
            },
        ]
        self.temp_passes = [
            {
                "id": generate_id("pass"),
                "login": "student",
                "zone": "Лаборатория 208",
                "start_at": now().isoformat(timespec="minutes"),
                "end_at": student_pass_end,
                "created_by": "admin",
                "status": "active",
            }
        ]
        self.requests = [
            {
                "id": generate_id("req"),
                "user_login": "student",
                "user_name": "Student",
                "user_role": "student",
                "zone": "Деканат",
                "reason": "Нужен временный доступ для встречи",
                "requested_until": (now() + timedelta(hours=2)).isoformat(timespec="minutes"),
                "target_role": "admin",
                "status": "pending",
                "comment": "",
                "created_at": now_str(),
                "reviewed_at": "",
                "reviewed_by": "",
            }
        ]
        self.logs = []
        self.save_all()

    def cleanup_expired_passes(self) -> None:
        changed = False
        current = now()
        for item in self.temp_passes:
            try:
                end_at = datetime.fromisoformat(item.get("end_at", ""))
                if item.get("status") == "active" and end_at <= current:
                    item["status"] = "expired"
                    changed = True
            except Exception:
                pass
        if changed:
            self.save_all()

    def authenticate(self, login: str, password: str) -> Optional[Dict[str, Any]]:
        for user in self.users:
            if user.get("login") == login and user.get("password") == password:
                return user
        return None

    def get_user(self, login: str) -> Optional[Dict[str, Any]]:
        for user in self.users:
            if user.get("login") == login:
                return user
        return None

    def add_user(self, login: str, password: str, name: str, role: str, zones: List[str]) -> Tuple[bool, str]:
        if not login or not password or not name:
            return False, "Заполни логин, пароль и имя"
        if role not in ROLES:
            return False, "Неверная роль"
        if any(u.get("login") == login for u in self.users):
            return False, "Такой логин уже существует"
        self.users.append(
            {
                "login": login,
                "password": password,
                "name": name,
                "role": role,
                "zones": zones,
                "signature": [],
                "photo_path": "",
                "created_at": now_str(),
            }
        )
        self.save_all()
        return True, "Пользователь добавлен"


    def add_log(self, item: Dict[str, Any]) -> None:
        self.logs.insert(0, item)
        self.logs = self.logs[:800]
        self.save_all()

    def add_request(self, user_login: str, zone: str, reason: str, requested_until: str, target_role: str = "admin") -> Tuple[bool, str]:
        user = self.get_user(user_login)
        if user is None:
            return False, "Пользователь не найден"
        if not zone or not reason.strip() or not requested_until:
            return False, "Заполни все поля заявки"
        if target_role not in ["admin", "teacher"]:
            target_role = "admin"
        self.requests.insert(
            0,
            {
                "id": generate_id("req"),
                "user_login": user_login,
                "user_name": user.get("name", ""),
                "user_role": user.get("role", "student"),
                "zone": zone,
                "reason": reason.strip(),
                "requested_until": requested_until,
                "target_role": target_role,
                "status": "pending",
                "comment": "",
                "created_at": now_str(),
                "reviewed_at": "",
                "reviewed_by": "",
            },
        )
        self.save_all()
        return True, "Заявка отправлена"

    def review_request(self, request_id: str, approved: bool, reviewer: str, comment: str = "") -> Tuple[bool, str]:
        for item in self.requests:
            if item.get("id") == request_id:
                if item.get("status") != "pending":
                    return False, "Эта заявка уже обработана"
                item["status"] = "approved" if approved else "rejected"
                item["comment"] = comment.strip()
                item["reviewed_at"] = now_str()
                item["reviewed_by"] = reviewer
                if approved:
                    self.temp_passes.insert(
                        0,
                        {
                            "id": generate_id("pass"),
                            "login": item.get("user_login"),
                            "zone": item.get("zone"),
                            "start_at": now().isoformat(timespec="minutes"),
                            "end_at": item.get("requested_until"),
                            "created_by": reviewer,
                            "status": "active",
                        },
                    )
                self.save_all()
                return True, "Заявка обработана"
        return False, "Заявка не найдена"

    def get_active_passes_for_user(self, login: str) -> List[Dict[str, Any]]:
        self.cleanup_expired_passes()
        return [p for p in self.temp_passes if p.get("login") == login and p.get("status") == "active"]

    def get_all_user_zones(self, login: str) -> List[str]:
        user = self.get_user(login)
        zones = list(user.get("zones", []) if user else [])
        for temp in self.get_active_passes_for_user(login):
            zone = temp.get("zone")
            if zone and zone not in zones:
                zones.append(zone)
        return zones

    def upcoming_expiry_notice(self, login: str) -> Optional[str]:
        current = now()
        notices: List[Tuple[timedelta, str]] = []
        for temp in self.get_active_passes_for_user(login):
            try:
                end_at = datetime.fromisoformat(temp.get("end_at", ""))
                delta = end_at - current
                if timedelta(seconds=0) < delta <= timedelta(minutes=10):
                    notices.append((delta, temp.get("zone", "зона")))
            except Exception:
                continue
        if not notices:
            return None
        notices.sort(key=lambda x: x[0])
        zone = notices[0][1]
        return f"Доступ истекает через 10 минут: {zone}"


    def _safe_remove_file(self, path: str) -> None:
        try:
            if path and os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass

    def delete_user(self, login: str) -> Tuple[bool, str]:
        if not login:
            return False, "Пользователь не выбран"
        if login == "admin":
            return False, "Главного админа удалять нельзя"

        user = self.get_user(login)
        if user is None:
            return False, "Пользователь не найден"

        self._safe_remove_file(user.get("photo_path", ""))
        self.users = [u for u in self.users if u.get("login") != login]
        self.requests = [r for r in self.requests if r.get("user_login") != login]
        self.temp_passes = [p for p in self.temp_passes if p.get("login") != login]
        self.save_all()
        return True, f"Пользователь {login} удалён"

    def delete_users(self, logins: List[str]) -> Tuple[bool, str]:
        cleaned = []
        for login in logins:
            if login and login != "admin" and login not in cleaned:
                cleaned.append(login)
        if not cleaned:
            return False, "Нет пользователей для удаления"

        removed = 0
        for login in cleaned:
            ok, _ = self.delete_user(login)
            if ok:
                removed += 1

        if removed == 0:
            return False, "Пользователи не удалены"
        return True, f"Удалено пользователей: {removed}"

    def clear_users(self) -> Tuple[bool, str]:
        removable = [u for u in self.users if u.get("login") != "admin"]
        if not removable:
            return False, "Нет пользователей для удаления"

        for user in removable:
            self._safe_remove_file(user.get("photo_path", ""))

        self.users = [u for u in self.users if u.get("login") == "admin"]
        self.requests = []
        self.temp_passes = []
        self.save_all()
        return True, "Все пользователи, кроме admin, удалены"

    def clear_logs(self) -> Tuple[bool, str]:
        self.logs = []
        self.save_all()
        return True, "Журнал очищен"

    def clear_requests(self) -> Tuple[bool, str]:
        self.requests = []
        self.save_all()
        return True, "Заявки очищены"

    def clear_temp_passes(self) -> Tuple[bool, str]:
        self.temp_passes = []
        self.save_all()
        return True, "Временные пропуска очищены"

    def clear_captures(self) -> Tuple[bool, str]:
        removed = 0
        if os.path.isdir(CAPTURE_DIR):
            for filename in os.listdir(CAPTURE_DIR):
                fpath = os.path.join(CAPTURE_DIR, filename)
                if os.path.isfile(fpath):
                    try:
                        os.remove(fpath)
                        removed += 1
                    except Exception:
                        pass

        for user in self.users:
            user["photo_path"] = ""

        for log in self.logs:
            log["photo_path"] = ""

        self.save_all()
        return True, f"Фото очищены: {removed}"
    def metrics(self) -> Dict[str, Any]:
        self.cleanup_expired_passes()
        total = max(1, len(self.logs))
        allowed = sum(1 for x in self.logs if x.get("result") == "allowed")
        denied = sum(1 for x in self.logs if x.get("result") == "denied")
        spoof = sum(1 for x in self.logs if x.get("result") == "spoof")
        unknown = sum(1 for x in self.logs if x.get("result") == "unknown")
        suspicious = sum(1 for x in self.logs if x.get("suspicious"))
        pending_requests = sum(1 for x in self.requests if x.get("status") == "pending")
        active_zones = len({x.get("zone") for x in self.logs if x.get("result") == "allowed" and x.get("zone")})
        far = round((unknown / total) * 100, 1)
        frr = round((denied / total) * 100, 1)
        return {
            "total": len(self.logs),
            "allowed": allowed,
            "denied": denied,
            "spoof": spoof,
            "unknown": unknown,
            "suspicious": suspicious,
            "pending_requests": pending_requests,
            "active_zones": active_zones,
            "far": far,
            "frr": frr,
        }

    def day_parts(self) -> Dict[str, int]:
        counts = {"morning": 0, "day": 0, "evening": 0}
        for item in self.logs:
            counts[classify_time_of_day(item.get("time", now_str()))] += 1
        return counts


class AccessApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.store = DataStore()
        self.current_user: Optional[Dict[str, Any]] = None
        self.current_tab = "home"
        self.sidebar_buttons: Dict[str, ft.Container] = {}

        self.camera_running = False
        self.current_frame = None
        self.cap = None

        self.live_camera_image = ft.Image(src_base64=EMPTY_IMAGE_BASE64, fit="cover", border_radius=18, width=360, height=240)

        self.page_container = ft.Container(expand=True)
        self.login_message = ft.Text("", color=RED, size=13)
        self.scan_status = ft.Text("Готово к сканированию", size=22, weight=ft.FontWeight.W_700, color=TEXT)
        self.scan_sub = ft.Text("Камера работает в реальном времени. Нажми кнопку для захвата.", color=MUTED, size=13)
        self.scan_chip = self.pill("ОЖИДАНИЕ", BLUE_SOFT, BLUE)

        self.page.title = "СКУД с распознаванием лица"
        self.page.bgcolor = BG
        self.page.padding = 0
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.scroll = ft.ScrollMode.AUTO

        # Android ignores desktop window properties, but Windows preview can still use them.
        # No large minimum sizes here, so the same project can open on mobile screens.
        try:
            self.page.window_width = 390
            self.page.window_height = 844
            self.page.window_resizable = True
        except Exception:
            pass

        self.build_login_view()


    def txt(self, value: str, size: int = 14, color: str = TEXT, weight=ft.FontWeight.W_500):
        return ft.Text(value, size=size, color=color, weight=weight)

    def pill(self, text: str, bg: str, color: str) -> ft.Container:
        return ft.Container(
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
            bgcolor=bg,
            border_radius=999,
            content=ft.Text(text, color=color, size=12, weight=ft.FontWeight.W_700),
        )

    def card(self, title: str, body: ft.Control, expand: bool = False, width: Optional[int] = None) -> ft.Container:
        return ft.Container(
            expand=expand,
            width=width,
            bgcolor=SURFACE,
            border_radius=24,
            border=ft.border.all(1, BORDER),
            padding=24,
            content=ft.Column([
                self.txt(title, size=18, weight=ft.FontWeight.W_700),
                ft.Container(height=12),
                body,
            ], spacing=0),
        )

    def metric_card(self, title: str, value: str, accent: str, soft: str) -> ft.Container:
        return ft.Container(
            expand=True,
            bgcolor=SURFACE,
            border_radius=22,
            border=ft.border.all(1, BORDER),
            padding=18,
            content=ft.Column([
                self.txt(title, size=13, color=MUTED),
                ft.Container(height=6),
                self.txt(value, size=28, color=accent, weight=ft.FontWeight.W_700),
                ft.Container(height=10),
                ft.Container(height=8, border_radius=999, bgcolor=soft),
            ], spacing=0),
        )

    def progress_row(self, label: str, value: int, max_value: int, accent: str, soft: str) -> ft.Control:
        ratio = 0 if max_value <= 0 else value / max_value
        return ft.Column([
            ft.Row([
                self.txt(label, color=MUTED),
                ft.Container(expand=True),
                self.txt(str(value), color=accent, weight=ft.FontWeight.W_700),
            ]),
            ft.ProgressBar(value=ratio, bgcolor=soft, color=accent, bar_height=10),
        ], spacing=4)

    def image_box(self, path: str, w: int = 80, h: int = 64) -> ft.Control:
        uri = image_file_to_data_uri(path)
        if uri:
            return ft.Image(src=uri, width=w, height=h, fit="cover", border_radius=16)
        return ft.Container(width=w, height=h, border_radius=16, bgcolor=PRIMARY_SOFT, alignment=ft.Alignment(0, 0),
                            content=ft.Icon(icon("PERSON"), color=PRIMARY))

    def nav_item(self, key: str, label: str, icon: str) -> ft.Container:
        btn = ft.Container(
            border_radius=18,
            padding=12,
            ink=True,
            on_click=lambda e, k=key: self.switch_tab(k),
            content=ft.Row([
                ft.Icon(icon, color=MUTED, size=20),
                self.txt(label, size=14, color=MUTED, weight=ft.FontWeight.W_600),
            ], spacing=12),
        )
        self.sidebar_buttons[key] = btn
        return btn

    def top_notice(self) -> Optional[ft.Control]:
        if not self.current_user:
            return None
        note = self.store.upcoming_expiry_notice(self.current_user.get("login", ""))
        if not note:
            return None
        return ft.Container(
            bgcolor=AMBER_SOFT,
            border_radius=18,
            border=ft.border.all(1, "#FED7AA"),
            padding=14,
            content=ft.Row([
                ft.Icon(icon("NOTIFICATIONS_ACTIVE_ROUNDED"), color=AMBER),
                self.txt(note, color="#9A5B00", weight=ft.FontWeight.W_700),
            ]),
        )

    def snack(self, text: str, ok: bool = True) -> None:
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text(text),
            bgcolor=GREEN if ok else RED,
            duration=2200,
        )
        self.page.snack_bar.open = True
        self.page.update()

    def confirm_action(self, title: str, message: str, on_confirm) -> None:
        def close_dialog(e=None):
            try:
                dlg.open = False
                self.page.update()
            except Exception:
                pass

        def do_confirm(e=None):
            close_dialog()
            try:
                on_confirm()
            except Exception as ex:
                self.snack(f"Ошибка: {ex}", False)

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(title, weight=ft.FontWeight.W_700),
            content=ft.Text(message),
            actions=[
                ft.TextButton("Отмена", on_click=close_dialog),
                ft.ElevatedButton("Подтвердить", bgcolor=RED, color="white", on_click=do_confirm),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.dialog = dlg
        dlg.open = True
        self.page.update()

    def force_refresh_current_view(self, target_tab: Optional[str] = None) -> None:
        try:
            if self.page.dialog:
                self.page.dialog.open = False
        except Exception:
            pass
        try:
            self.page.dialog = None
        except Exception:
            pass
        self.store.cleanup_expired_passes()
        current = target_tab or self.current_tab or self.get_nav_items()[0][0]
        self.page.clean()
        self.build_shell()
        self.switch_tab(current)


    def build_login_view(self) -> None:
        self.page.clean()
        self.page.navigation_bar = None
        self.page.scroll = ft.ScrollMode.HIDDEN
        self.page.vertical_alignment = ft.MainAxisAlignment.CENTER
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

        self.login_field = ft.TextField(
            label="Логин",
            width=360,
            border_radius=18,
            filled=True,
            bgcolor=SURFACE_2,
            border_color=BORDER,
        )
        self.password_field = ft.TextField(
            label="Пароль",
            width=360,
            password=True,
            can_reveal_password=True,
            border_radius=18,
            filled=True,
            bgcolor=SURFACE_2,
            border_color=BORDER,
        )

        def do_login(e=None):
            user = self.store.authenticate(self.login_field.value.strip(), self.password_field.value.strip())
            if not user:
                self.login_message.value = "Неверный логин или пароль"
                self.page.update()
                return
            self.current_user = user
            self.login_message.value = ""
            self.build_shell()
            default_tab = self.get_nav_items()[0][0]
            self.switch_tab(default_tab)

        login_card = ft.Container(
            width=520,
            bgcolor=SURFACE,
            border_radius=34,
            border=ft.border.all(1, BORDER),
            padding=32,
            shadow=ft.BoxShadow(blur_radius=30, color="#16000000", offset=ft.Offset(0, 12)),
            content=ft.Column([
                ft.Container(
                    width=76,
                    height=76,
                    border_radius=24,
                    bgcolor=PRIMARY_SOFT,
                    alignment=ft.Alignment(0, 0),
                    content=ft.Icon(icon("FACE_RETOUCHING_NATURAL_ROUNDED"), color=PRIMARY, size=38),
                ),
                self.txt("FACE ACCESS SYSTEM", size=34, weight=ft.FontWeight.W_700),
                self.txt("Система контроля доступа с распознаванием лиц", size=15, color=MUTED),
                ft.Container(height=8),
                self.login_field,
                self.password_field,
                ft.ElevatedButton(
                    "Войти",
                    width=360,
                    height=56,
                    bgcolor=PRIMARY,
                    color="white",
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=18)),
                    on_click=do_login,
                ),
                self.login_message,
            ], spacing=14, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        )

        hero = ft.Container(
            expand=True,
            bgcolor=BG,
            content=ft.Row([
                ft.Container(expand=True),
                login_card,
                ft.Container(expand=True),
            ], alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )
        self.page.add(hero)
        self.page.update()

    def logout(self) -> None:
        self.stop_camera()

        self.current_user = None
        self.current_tab = "home"
        self.build_login_view()

    def get_nav_items(self) -> List[Tuple[str, str, str]]:
        role = self.current_user.get("role", "student") if self.current_user else "student"
        if role == "student":
            return [
                ("home", "Главная", icon("HOME_ROUNDED")),
                ("face", "Скан", icon("FACE_RETOUCHING_NATURAL_ROUNDED")),
                ("requests", "Заявки", icon("MARK_EMAIL_READ_OUTLINED")),
                ("history", "История", icon("RECEIPT_LONG_ROUNDED")),
                ("profile", "Профиль", icon("PERSON_ROUNDED")),
            ]
        if role == "teacher":
            return [
                ("home", "Главная", icon("HOME_ROUNDED")),
                ("face", "Скан", icon("FACE_RETOUCHING_NATURAL_ROUNDED")),
                ("requests", "Заявки", icon("RULE_ROUNDED")),
                ("history", "Журнал", icon("RECEIPT_LONG_ROUNDED")),
                ("students", "Студенты", icon("GROUPS_2_ROUNDED")),
            ]
        return [
            ("dashboard", "Главное", icon("DASHBOARD_ROUNDED")),
            ("requests", "Заявки", icon("APPROVAL_ROUNDED")),
            ("logs", "Журнал", icon("RECEIPT_LONG_ROUNDED")),
            ("analytics", "Аналитика", icon("INSERT_CHART_OUTLINED_ROUNDED")),
            ("database", "База", icon("SETTINGS_ROUNDED")),
        ]

    def build_shell(self) -> None:
        self.page.clean()
        self.page.vertical_alignment = ft.MainAxisAlignment.START
        self.page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
        self.page.scroll = ft.ScrollMode.HIDDEN

        user = self.current_user or {}
        role_label = {"student": "Студент", "teacher": "Преподаватель", "admin": "Админ"}.get(user.get("role"), "Пользователь")

        header = ft.Container(
            bgcolor=SURFACE,
            padding=ft.padding.symmetric(horizontal=24, vertical=16),
            border=ft.border.only(bottom=ft.BorderSide(1, BORDER)),
            content=ft.Row([
                ft.Row([
                    ft.Icon(icon("VERIFIED_USER_ROUNDED"), color=PRIMARY, size=28),
                    self.txt("Face Access", size=20, weight=ft.FontWeight.W_700),
                ], spacing=10),
                ft.Container(expand=True),
                ft.Row([
                    self.txt(f"{user.get('name', '')} ({role_label})", size=14, color=MUTED),
                    ft.IconButton(icon("LOGOUT_ROUNDED"), icon_color=RED, on_click=lambda e: self.logout())
                ])
            ])
        )

        content = ft.Container(
            expand=True,
            bgcolor=BG,
            padding=ft.padding.only(left=24, top=20, right=24, bottom=12),
            content=self.page_container,
        )

        nav_items = self.get_nav_items()
        destinations = []
        for key, label, nav_icon in nav_items:
            if NAV_BAR_DEST is not None:
                try:
                    destinations.append(NAV_BAR_DEST(icon=nav_icon, selected_icon=nav_icon, label=label))
                except TypeError:
                    destinations.append(NAV_BAR_DEST(icon=nav_icon, label=label))

        self.page.navigation_bar = ft.NavigationBar(
            destinations=destinations,
            selected_index=0,
            label_behavior=ft.NavigationBarLabelBehavior.ALWAYS_SHOW,
            height=78,
            bgcolor=SURFACE,
            indicator_color=PRIMARY_SOFT,
            elevation=8,
            on_change=lambda e: self.switch_tab(nav_items[e.control.selected_index][0]),
        )

        self.page.add(header, content)
        self.page.update()


    def update_sidebar(self) -> None:
        for key, btn in self.sidebar_buttons.items():
            active = key == self.current_tab
            btn.bgcolor = PRIMARY if active else SURFACE
            row = btn.content
            row.controls[0].color = "white" if active else MUTED
            row.controls[1].color = "white" if active else TEXT
            btn.border = ft.border.all(1, PRIMARY if active else BORDER)
        self.page.update()

    def start_camera(self):
        if not self.camera_running:
            self.camera_running = True
            threading.Thread(target=self.camera_loop, daemon=True).start()

    def stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def camera_loop(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            fallback_path = create_fallback_image("NO CAMERA FOUND")
            uri = image_file_to_data_uri(fallback_path)
            self.live_camera_image.src_base64 = None
            self.live_camera_image.src = uri
            try:
                self.live_camera_image.update()
            except Exception:
                pass
            self.camera_running = False
            return

        while self.camera_running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1) 
                self.current_frame = frame
                _, buffer = cv2.imencode('.jpg', frame)
                b64 = base64.b64encode(buffer).decode('utf-8')
                self.live_camera_image.src = "" 
                self.live_camera_image.src_base64 = b64 

                try:
                    self.live_camera_image.update()
                except Exception:
                    pass
            time.sleep(0.04) 

        if self.cap:
            self.cap.release()

    def switch_tab(self, key: str) -> None:
        self.store.cleanup_expired_passes()

        if key in ["face", "dashboard"]:
            self.start_camera()
        else:
            self.stop_camera()

        self.current_tab = key
        role = self.current_user.get("role", "student") if self.current_user else "student"

        if role == "student":
            mapping = {"home": self.student_home, "face": self.face_page, "requests": self.student_requests_page, "history": self.student_history_page, "profile": self.student_profile_page}
        elif role == "teacher":
            mapping = {"home": self.teacher_home, "face": self.face_page, "requests": self.teacher_requests_page, "history": self.teacher_history_page, "students": self.teacher_students_page}
        else:
            mapping = {
                "dashboard": self.admin_dashboard,
                "requests": self.admin_requests_page,
                "logs": self.admin_logs_page,
                "analytics": self.admin_analytics_page,
                "database": self.admin_database_page,
            }

        builder = mapping.get(key)
        if builder:
            notice = self.top_notice()
            body = builder()
            if notice:
                self.page_container.content = ft.Column([notice, ft.Container(height=14), body], expand=True, scroll=ft.ScrollMode.AUTO)
            else:
                self.page_container.content = body

        nav_items = self.get_nav_items()
        for i, (k, l, nav_icon) in enumerate(nav_items):
            if k == key and self.page.navigation_bar:
                self.page.navigation_bar.selected_index = i
                break
        self.page.update()
    
    def run_scan(self, zone: str) -> None:
        if self.current_frame is None:
            self.snack("Камера не готова или не найдена", False)
            return

        frame = self.current_frame.copy()
        user = self.current_user or {}
        zone = zone or "Главный вход"

        capture_path = save_frame(frame, "attempt")
        face = crop_biggest_face(frame)

        suspicious = False
        result = "unknown"
        reason = ""
        match_score = 0.0

        if face is None:
            suspicious = True
            reason = "Лицо не найдено в кадре"
        else:
            sig = make_signature(face)
            user_sig = user.get("signature", [])

            if not user_sig:
                result = "denied"
                reason = "В вашем профиле нет сохранённого лица"
            else:
                match_score = cosine_similarity(sig, user_sig)
                if match_score > 0.70: 
                    allowed_zones = self.store.get_all_user_zones(user.get("login", ""))
                    if zone not in allowed_zones:
                        result = "denied"
                        reason = f"Нет доступа в зону: {zone}"
                    else:
                        result = "allowed"
                        reason = f"Совпадение подтверждено, зона: {zone}"
                else:
                    suspicious = True
                    result = "spoof"
                    reason = "Лицо не совпадает с владельцем профиля"

        chip_map = {
            "allowed": ("ДОСТУП РАЗРЕШЁН", GREEN_SOFT, GREEN),
            "denied": ("ДОСТУП ОТКЛОНЁН", RED_SOFT, RED),
            "spoof": ("ПОДОЗРИТЕЛЬНАЯ ПОПЫТКА", AMBER_SOFT, AMBER),
            "unknown": ("ОШИБКА", BLUE_SOFT, BLUE),
        }
        chip_text, chip_bg, chip_color = chip_map[result]
        self.scan_chip = self.pill(chip_text, chip_bg, chip_color)
        self.scan_status.value = chip_text.capitalize()
        self.scan_status.color = chip_color
        self.scan_sub.value = f"{reason}. Совпадение: {match_score:.2f}"

        self.store.add_log(
            {
                "time": now_str(),
                "name": user.get("name", "Unknown"),
                "role": user.get("role", "-"),
                "login": user.get("login", "-"),
                "zone": zone,
                "result": result,
                "reason": reason,
                "photo_path": capture_path,
                "camera_used": True,
                "suspicious": suspicious,
                "confidence": round(match_score, 2),
            }
        )
        self.page.update()

    def face_page(self) -> ft.Control:
        zone_dropdown = ft.Dropdown(
            label="Зона доступа", value=ZONES[0], options=[ft.dropdown.Option(z) for z in ZONES],
            width=260, border_radius=16, filled=True, bgcolor=SURFACE_2, border_color=BORDER,
        )

        left = self.card(
            "Распознавание лица",
            ft.Column([
                ft.Container(
                    bgcolor=SURFACE_2,
                    border_radius=22,
                    padding=14,
                    content=ft.Container(height=270, alignment=ft.Alignment(0, 0), content=self.live_camera_image),
                ),
                ft.Container(height=14),
                ft.Row([
                    zone_dropdown,
                    ft.ElevatedButton(
                        "Проверить",
                        icon=icon("CAMERA_ALT_ROUNDED"), height=54, bgcolor=PRIMARY, color="white",
                        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=16)),
                        on_click=lambda e: self.run_scan(zone_dropdown.value),
                    ),
                ], wrap=True, spacing=12),
            ], spacing=0),
            expand=True,
        )

        result_rows: List[ft.Control] = [
            self.scan_chip,
            ft.Container(height=18),
            self.scan_status,
            ft.Container(height=8),
            self.scan_sub,
            ft.Container(height=18),
            ft.Container(
                padding=16,
                border_radius=18,
                bgcolor=SURFACE_2,
                content=ft.Column([
                    self.txt("Статус проверки", size=14, color=MUTED),
                    ft.Container(height=8),
                    self.txt("После сканирования здесь появится результат доступа", size=14, color=TEXT),
                ], spacing=0),
            ),
        ]

        right = self.card("Результат", ft.Column(result_rows, spacing=0), width=390)
        return ft.Row([left, right], spacing=18, expand=True, vertical_alignment=ft.CrossAxisAlignment.START)

    def request_card(self, item: Dict[str, Any], can_review: bool) -> ft.Control:
        status = item.get("status", "pending")
        color_map = {
            "pending": (BLUE_SOFT, BLUE, "ОЖИДАЕТ"),
            "approved": (GREEN_SOFT, GREEN, "ОДОБРЕНО"),
            "rejected": (RED_SOFT, RED, "ОТКЛОНЕНО"),
        }
        chip_bg, chip_color, chip_text = color_map.get(status, (BLUE_SOFT, BLUE, status.upper()))
        body_controls: List[ft.Control] = [
            ft.Row([
                ft.Column([
                    self.txt(item.get("user_name", "Пользователь"), size=15, weight=ft.FontWeight.W_700),
                    self.txt(f"Роль: {item.get('user_role', '-')}", size=12, color=MUTED),
                    self.txt(f"Зона: {item.get('zone', '-')}", size=12, color=PRIMARY),
                    self.txt(f"Кому: {'Админ' if item.get('target_role', 'admin') == 'admin' else 'Преподаватель'}", size=12, color=MUTED),
                    self.txt(f"До: {fmt_dt(item.get('requested_until', ''))}", size=12, color=MUTED),
                    self.txt(f"Причина: {item.get('reason', '')}", size=12, color=TEXT),
                ], spacing=4, expand=True),
                self.pill(chip_text, chip_bg, chip_color),
            ]),
        ]
        if item.get("comment"):
            body_controls += [ft.Container(height=6), self.txt(f"Комментарий: {item.get('comment')}", size=12, color=chip_color)]
        if item.get("reviewed_by"):
            body_controls += [self.txt(f"Проверил: {item.get('reviewed_by')} • {item.get('reviewed_at')}", size=12, color=MUTED)]
        if can_review and status == "pending":
            comment_field = ft.TextField(label="Комментарий", multiline=True, min_lines=1, max_lines=3, border_radius=14, filled=True, bgcolor=SURFACE_2)
            actions = ft.Row([
                ft.ElevatedButton(
                    "Одобрить",
                    bgcolor=GREEN,
                    color="white",
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=14)),
                    on_click=lambda e, rid=item["id"], field=comment_field: self.handle_review(rid, True, field.value or ""),
                ),
                ft.ElevatedButton(
                    "Отклонить",
                    bgcolor=RED,
                    color="white",
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=14)),
                    on_click=lambda e, rid=item["id"], field=comment_field: self.handle_review(rid, False, field.value or "Без комментария"),
                ),
            ], spacing=10)
            body_controls += [ft.Container(height=10), comment_field, ft.Container(height=8), actions]
        return ft.Container(bgcolor=SURFACE, border_radius=20, border=ft.border.all(1, BORDER), padding=16, content=ft.Column(body_controls, spacing=0))

    def handle_review(self, request_id: str, approved: bool, comment: str) -> None:
        reviewer = self.current_user.get("login", "admin") if self.current_user else "admin"
        ok, msg = self.store.review_request(request_id, approved, reviewer, comment)
        self.snack(msg, ok)
        self.switch_tab(self.current_tab)

    def logs_list(self, items: List[Dict[str, Any]]) -> ft.Control:
        if not items:
            return self.card("Журнал", self.txt("Событий пока нет", color=MUTED), expand=True)
        rows: List[ft.Control] = []
        for item in items[:80]:
            result = item.get("result", "unknown")
            if result == "allowed":
                bg, color, label = GREEN_SOFT, GREEN, "РАЗРЕШЕНО"
            elif result == "denied":
                bg, color, label = RED_SOFT, RED, "ОТКАЗ"
            elif result == "spoof":
                bg, color, label = AMBER_SOFT, AMBER, "СПУФИНГ"
            else:
                bg, color, label = BLUE_SOFT, BLUE, "НЕИЗВЕСТНО"
            rows.append(
                ft.Container(
                    bgcolor=SURFACE,
                    border_radius=20,
                    border=ft.border.all(1, BORDER),
                    padding=14,
                    content=ft.Row([
                        self.image_box(item.get("photo_path", ""), 92, 72),
                        ft.Column([
                            self.txt(item.get("name", "Неизвестно"), size=15, weight=ft.FontWeight.W_700),
                            self.txt(f"{item.get('time', '')} • {item.get('zone', '')}", size=12, color=MUTED),
                            self.txt(item.get("reason", ""), size=12, color=color),
                            self.txt(f"Роль: {item.get('role', '-')} • confidence: {item.get('confidence', 0)}", size=11, color=MUTED),
                        ], spacing=3, expand=True),
                        self.pill(label, bg, color),
                    ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                )
            )
        return ft.Column(rows, spacing=10, expand=True, scroll=ft.ScrollMode.AUTO)

    def student_home(self) -> ft.Control:
        login = self.current_user.get("login", "")
        active = self.store.get_active_passes_for_user(login)
        user_logs = [x for x in self.store.logs if x.get("login") == login]
        pending = [x for x in self.store.requests if x.get("user_login") == login and x.get("status") == "pending"]
        zones = self.store.get_all_user_zones(login)

        zone_controls = [
            ft.Container(
                bgcolor=SURFACE_2,
                border_radius=18,
                padding=14,
                content=ft.Row([
                    self.txt(zone, size=14),
                    ft.Container(expand=True),
                    self.pill("Есть доступ", GREEN_SOFT, GREEN),
                ]),
            ) for zone in zones
        ] or [self.txt("Нет зон", color=MUTED)]

        temp_controls = [
            ft.Container(
                bgcolor=SURFACE_2,
                border_radius=18,
                padding=14,
                content=ft.Row([
                    self.txt(p.get("zone", "—"), size=13),
                    ft.Container(expand=True),
                    self.pill(f"до {fmt_dt(p.get('end_at', ''))}", PRIMARY_SOFT, PRIMARY),
                ]),
            ) for p in active
        ] or [self.txt("Нет активных временных пропусков", color=MUTED)]

        return ft.Column([
            ft.Row([
                self.metric_card("Мои события", str(len(user_logs)), BLUE, BLUE_SOFT),
                self.metric_card("Активные заявки", str(len(pending)), AMBER, AMBER_SOFT),
                self.metric_card("Временные пропуска", str(len(active)), PRIMARY, PRIMARY_SOFT),
            ], spacing=14),
            ft.Container(height=16),
            ft.Row([
                self.card("Мои зоны доступа", ft.Column(zone_controls, spacing=10), expand=True),
                self.card("Временный доступ", ft.Column(temp_controls, spacing=10), expand=True),
            ], spacing=16),
        ], expand=True, scroll=ft.ScrollMode.AUTO)

    def student_requests_page(self) -> ft.Control:
        login = self.current_user.get("login", "")
        reason = ft.TextField(label="Причина заявки", multiline=True, min_lines=3, max_lines=4, border_radius=16, filled=True, bgcolor=SURFACE_2)
        zone = ft.Dropdown(label="Зона", value=ZONES[3], options=[ft.dropdown.Option(z) for z in ZONES], width=250, border_radius=16, filled=True, bgcolor=SURFACE_2)
        recipient = ft.Dropdown(
            label="Кому отправить",
            value="admin",
            width=220,
            options=[ft.dropdown.Option("admin", "Админ"), ft.dropdown.Option("teacher", "Преподаватель")],
            border_radius=16, filled=True, bgcolor=SURFACE_2,
        )
        until = ft.TextField(label="До какого времени", value=(now() + timedelta(hours=2)).isoformat(timespec="minutes"), width=260, border_radius=16, filled=True, bgcolor=SURFACE_2)

        def submit(e):
            ok, msg = self.store.add_request(login, zone.value, reason.value or "", until.value or "", recipient.value or "admin")
            self.snack(msg, ok)
            if ok:
                self.switch_tab("requests")

        my_requests = [x for x in self.store.requests if x.get("user_login") == login]
        return ft.Column([
            self.card(
                "Подать заявку на доступ",
                ft.Column([
                    ft.Row([zone, recipient, until], wrap=True, spacing=12),
                    ft.Container(height=12),
                    ft.Container(
                        bgcolor=SURFACE_2,
                        border_radius=18,
                        padding=16,
                        content=reason,
                    ),
                    ft.Container(height=12),
                    ft.ElevatedButton(
                        "Отправить заявку",
                        icon=icon("SEND_ROUNDED"),
                        bgcolor=PRIMARY,
                        color="white",
                        height=52,
                        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=16)),
                        on_click=submit,
                    ),
                ], spacing=0),
                expand=True,
            ),
            ft.Container(height=16),
            self.card("Мои заявки", ft.Column([self.request_card(x, False) for x in my_requests] or [self.txt("Заявок пока нет", color=MUTED)], spacing=10), expand=True),
        ], expand=True, scroll=ft.ScrollMode.AUTO)

    def student_history_page(self) -> ft.Control:
        login = self.current_user.get("login", "")
        items = [x for x in self.store.logs if x.get("login") == login]
        return self.logs_list(items)

    def student_profile_page(self) -> ft.Control:
        user = self.current_user or {}
        active_passes = self.store.get_active_passes_for_user(user.get("login", ""))
        pass_controls = [
            ft.Container(
                bgcolor=SURFACE_2,
                border_radius=18,
                padding=14,
                content=ft.Row([
                    self.txt(p.get("zone", "—")),
                    ft.Container(expand=True),
                    self.pill(f"до {fmt_dt(p.get('end_at', ''))}", PRIMARY_SOFT, PRIMARY),
                ]),
            ) for p in active_passes
        ] or [self.txt("Сейчас нет активных временных пропусков", color=MUTED)]

        return ft.Row([
            self.card(
                "Профиль",
                ft.Column([
                    self.image_box(user.get("photo_path", ""), 120, 120),
                    ft.Container(height=12),
                    self.txt(user.get("name", "Пользователь"), size=24, weight=ft.FontWeight.W_700),
                    self.txt(f"Роль: {user.get('role', 'student')}", color=MUTED),
                    self.txt(f"Логин: {user.get('login', '-')}", color=MUTED),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
                width=340,
            ),
            self.card(
                "Активные пропуска",
                ft.Column(pass_controls, spacing=10),
                expand=True,
            ),
        ], spacing=16, expand=True, vertical_alignment=ft.CrossAxisAlignment.START)

    def teacher_home(self) -> ft.Control:
        zones = self.store.get_all_user_zones(self.current_user.get("login", ""))
        zone_cards = [
            ft.Container(
                bgcolor=SURFACE_2,
                border_radius=18,
                padding=16,
                content=ft.Row([
                    self.txt(z, size=14),
                    ft.Container(expand=True),
                    self.pill("Есть доступ", GREEN_SOFT, GREEN),
                ]),
            ) for z in zones
        ] or [self.txt("Нет доступных зон", color=MUTED)]

        return ft.Column([
            self.card("Мои зоны", ft.Column(zone_cards, spacing=10), expand=True),
        ], expand=True, scroll=ft.ScrollMode.AUTO)

    def teacher_requests_page(self) -> ft.Control:
        items = [x for x in self.store.requests if x.get("target_role", "admin") == "teacher"]
        return ft.Column([self.request_card(x, x.get("status") == "pending") for x in items] or [self.txt("Заявок нет", color=MUTED)], spacing=10, expand=True, scroll=ft.ScrollMode.AUTO)

    def teacher_history_page(self) -> ft.Control:
        student_logins = [u.get("login") for u in self.store.users if u.get("role") == "student"]
        items = [x for x in self.store.logs if x.get("login") in [self.current_user.get("login")] + student_logins]
        return ft.Column([
            self.card("Журнал", self.txt("Здесь показаны проверки студентов и ваши записи.", color=MUTED), expand=True),
            ft.Container(height=14),
            self.logs_list(items),
        ], expand=True, scroll=ft.ScrollMode.AUTO)

    def teacher_students_page(self) -> ft.Control:
        students = [u for u in self.store.users if u.get("role") == "student"]
        cards: List[ft.Control] = []
        for student in students:
            zones = self.store.get_all_user_zones(student.get("login", ""))
            cards.append(
                ft.Container(
                    bgcolor=SURFACE,
                    border_radius=20,
                    border=ft.border.all(1, BORDER),
                    padding=16,
                    content=ft.Row([
                        self.image_box(student.get("photo_path", ""), 72, 72),
                        ft.Column([
                            self.txt(student.get("name", ""), size=15, weight=ft.FontWeight.W_700),
                            self.txt(student.get("login", ""), size=12, color=MUTED),
                            self.txt(", ".join(zones), size=12, color=PRIMARY),
                        ], spacing=4, expand=True),
                        self.pill("Студент", PRIMARY_SOFT, PRIMARY),
                    ]),
                )
            )
        return ft.Column(cards or [self.txt("Студентов нет", color=MUTED)], spacing=10, expand=True, scroll=ft.ScrollMode.AUTO)

    def admin_dashboard(self) -> ft.Control:
        metrics = self.store.metrics()
        users = self.store.users
        recent_users = users[:6]

        add_login = ft.TextField(label="Логин", width=170, border_radius=14, filled=True, bgcolor=SURFACE_2)
        add_pass = ft.TextField(label="Пароль", width=170, border_radius=14, filled=True, bgcolor=SURFACE_2)
        add_name = ft.TextField(label="Имя", width=190, border_radius=14, filled=True, bgcolor=SURFACE_2)
        add_role = ft.Dropdown(label="Роль", value="student", width=170, options=[ft.dropdown.Option(r) for r in ROLES], border_radius=14, filled=True, bgcolor=SURFACE_2)
        zone_checks = [ft.Checkbox(label=z, value=(z in ROLE_DEFAULT_ZONES["student"])) for z in ZONES]
        check_zone_dropdown = ft.Dropdown(label="Зона для теста", value=ZONES[0], options=[ft.dropdown.Option(z) for z in ZONES], width=220, border_radius=14, filled=True, bgcolor=SURFACE_2)

        def role_change(e):
            defaults = ROLE_DEFAULT_ZONES.get(add_role.value, [])
            for cb in zone_checks:
                cb.value = cb.label in defaults
            self.page.update()

        add_role.on_change = role_change

        def add_user_with_photo(e):
            if self.current_frame is None:
                self.snack("Сначала включите камеру и покажите лицо", False)
                return
            frame = self.current_frame.copy()
            face = crop_biggest_face(frame)
            if face is None:
                self.snack("Лицо не найдено, встаньте ровно", False)
                return

            zones = [cb.label for cb in zone_checks if cb.value]
            ok, msg = self.store.add_user(add_login.value.strip(), add_pass.value.strip(), add_name.value.strip(), add_role.value, zones)

            if ok:
                user = self.store.get_user(add_login.value.strip())
                user["signature"] = make_signature(face)
                user["photo_path"] = save_frame(frame, f"profile_{add_login.value.strip()}")
                self.store.save_all()
                self.snack("Пользователь и его лицо успешно сохранены!", True)
                self.force_refresh_current_view("database")
            else:
                self.snack(msg, False)

        def add_user_without_photo(e):
            zones = [cb.label for cb in zone_checks if cb.value]
            ok, msg = self.store.add_user(add_login.value.strip(), add_pass.value.strip(), add_name.value.strip(), add_role.value, zones)
            self.store = DataStore()
            self.snack("Пользователь сохранён без фото" if ok else msg, ok)
            if ok:
                self.force_refresh_current_view("database")

        camera_card = self.card(
            "Распознавание лица",
            ft.Column([
                ft.Container(
                    bgcolor=SURFACE_2,
                    border_radius=22,
                    padding=14,
                    content=ft.Container(height=300, alignment=ft.Alignment(0, 0), content=self.live_camera_image),
                ),
                ft.Container(height=14),
                ft.Row([
                    check_zone_dropdown,
                    ft.ElevatedButton(
                        "Проверить лицо",
                        icon=icon("CAMERA_ALT_ROUNDED"),
                        bgcolor=PRIMARY,
                        color="white",
                        height=52,
                        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=16)),
                        on_click=lambda e: self.run_scan(check_zone_dropdown.value),
                    ),
                ], wrap=True, spacing=12),
                ft.Container(height=12),
                self.scan_chip,
                ft.Container(height=8),
                self.scan_sub,
            ], spacing=0),
            expand=True,
        )

        register_card = self.card(
            "Регистрация пользователя",
            ft.Column([
                ft.Row([add_login, add_pass, add_name, add_role], wrap=True, spacing=10),
                ft.Container(height=10),
                ft.Container(
                    bgcolor=SURFACE_2,
                    border_radius=18,
                    padding=16,
                    content=ft.Column([
                        self.txt("Зоны доступа", size=14, color=MUTED),
                        ft.Container(height=8),
                        ft.Row(zone_checks, wrap=True, spacing=8),
                    ], spacing=0),
                ),
                ft.Container(height=14),
                ft.Row([
                    ft.ElevatedButton(
                        "Сохранить с фото",
                        icon=icon("PERSON_ADD_ALT_1_ROUNDED"),
                        bgcolor=GREEN,
                        color="white",
                        height=52,
                        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=16)),
                        on_click=add_user_with_photo,
                    ),
                    ft.OutlinedButton(
                        "Сохранить без фото",
                        icon=icon("PERSON_ADD"),
                        height=52,
                        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=16)),
                        on_click=add_user_without_photo,
                    ),
                ], wrap=True, spacing=12),
            ], spacing=0),
            expand=True,
        )

        rows = [
            ft.Container(
                bgcolor=SURFACE_2, border_radius=18, padding=12,
                content=ft.Row([
                    self.image_box(u.get("photo_path", ""), 60, 60),
                    ft.Column([
                        self.txt(u.get("name", ""), size=14, weight=ft.FontWeight.W_700),
                        self.txt(f"{u.get('login', '')} • {u.get('role', '')}", size=12, color=MUTED),
                    ], spacing=3, expand=True),
                    ft.IconButton(icon("DELETE_OUTLINE"), icon_color=RED, on_click=lambda e, l=u.get("login"): self.admin_delete_single_user_confirm(l)),
                ]),
            ) for u in recent_users
        ]

        return ft.Column([
            ft.Row([
                self.metric_card("Пользователи", str(len(users)), PRIMARY, PRIMARY_SOFT),
                self.metric_card("Активные зоны", str(metrics["active_zones"]), GREEN, GREEN_SOFT),
                self.metric_card("Ожидают заявки", str(metrics["pending_requests"]), AMBER, AMBER_SOFT),
                self.metric_card("Подозрительные", str(metrics["suspicious"]), RED, RED_SOFT),
            ], spacing=14),
            ft.Container(height=16),
            ft.Row([camera_card, register_card], spacing=16, vertical_alignment=ft.CrossAxisAlignment.START),
            ft.Container(height=16),
            self.card("Последние пользователи", ft.Row(rows or [self.txt("Нет пользователей", color=MUTED)], wrap=True, spacing=10)),
        ], expand=True, scroll=ft.ScrollMode.AUTO)


    def admin_database_page(self) -> ft.Control:
        role_state = {"value": "student"}
        users_list = ft.Column(spacing=10)
        all_users_list = ft.Column(spacing=10)
        checkbox_login_map: Dict[int, str] = {}

        students_btn = ft.ElevatedButton(
            "Студенты",
            height=44,
            on_click=lambda e: switch_role("student"),
        )
        teachers_btn = ft.ElevatedButton(
            "Преподаватели",
            height=44,
            on_click=lambda e: switch_role("teacher"),
        )

        def styled_role_button(btn: ft.ElevatedButton, active: bool) -> None:
            btn.bgcolor = PRIMARY if active else SURFACE
            btn.color = "white" if active else TEXT
            btn.style = ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=14))

        def visible_users_by_role(selected_role: str) -> List[Dict[str, Any]]:
            return [
                u for u in self.store.users
                if u.get("login") != "admin" and u.get("role") == selected_role
            ]

        def rebuild_all_users_list() -> None:
            all_users_list.controls.clear()
            users = [u for u in self.store.users if u.get("login") != "admin"]
            if not users:
                all_users_list.controls.append(
                    ft.Container(
                        padding=14,
                        border_radius=14,
                        bgcolor=SURFACE_2,
                        border=ft.border.all(1, BORDER),
                        content=self.txt("В базе пока нет пользователей", color=MUTED),
                    )
                )
                return

            for u in users:
                zones = ", ".join(u.get("zones", [])[:3])
                if len(u.get("zones", [])) > 3:
                    zones += "..."
                role_label = "Студент" if u.get("role") == "student" else ("Преподаватель" if u.get("role") == "teacher" else "Админ")
                role_bg = PRIMARY_SOFT if u.get("role") == "student" else GREEN_SOFT
                role_color = PRIMARY if u.get("role") == "student" else GREEN
                all_users_list.controls.append(
                    ft.Container(
                        padding=14,
                        border_radius=16,
                        bgcolor=SURFACE,
                        border=ft.border.all(1, BORDER),
                        content=ft.Column([
                            ft.Row([
                                self.txt(u.get("name", "Без имени"), size=15, weight=ft.FontWeight.W_700),
                                ft.Container(expand=True),
                                ft.Container(
                                    padding=ft.padding.symmetric(horizontal=10, vertical=6),
                                    border_radius=999,
                                    bgcolor=role_bg,
                                    content=self.txt(role_label, size=11, weight=ft.FontWeight.W_700, color=role_color),
                                ),
                            ]),
                            ft.Container(height=8),
                            self.txt(f"Логин: {u.get('login', '—')}", size=12, color=MUTED),
                            self.txt(f"Пароль: {u.get('password', '—')}", size=12, color=MUTED),
                            self.txt(f"Зоны: {zones or '—'}", size=12, color=MUTED),
                        ], spacing=2),
                    )
                )

        def rebuild_user_list() -> None:
            selected_role = role_state["value"]
            filtered_users = visible_users_by_role(selected_role)
            checkbox_login_map.clear()
            users_list.controls.clear()

            if not filtered_users:
                users_list.controls.append(
                    ft.Container(
                        padding=16,
                        border_radius=14,
                        bgcolor=SURFACE_2,
                        border=ft.border.all(1, BORDER),
                        content=self.txt("В этой категории пока нет пользователей", color=MUTED),
                    )
                )
            else:
                for u in filtered_users:
                    cb = ft.Checkbox(
                        label=f"{u.get('name', '')} ({u.get('login', '')})",
                        value=False,
                    )
                    checkbox_login_map[id(cb)] = u.get("login", "")
                    users_list.controls.append(
                        ft.Container(
                            bgcolor=SURFACE,
                            border_radius=14,
                            padding=ft.padding.only(left=14, top=10, right=14, bottom=10),
                            border=ft.border.all(1, BORDER),
                            content=cb,
                        )
                    )
            styled_role_button(students_btn, role_state["value"] == "student")
            styled_role_button(teachers_btn, role_state["value"] == "teacher")

        def switch_role(role: str) -> None:
            role_state["value"] = role
            rebuild_user_list()
            self.page.update()

        rebuild_all_users_list()
        rebuild_user_list()

        database_list_block = self.card(
            "База пользователей",
            ft.Column([
                self.txt("Все студенты и преподаватели с логином, паролем и зонами доступа", size=13, color=MUTED),
                ft.Container(height=12),
                all_users_list,
            ], spacing=0),
            expand=True,
        )

        user_block = self.card(
            "Удаление пользователей",
            ft.Column([
                self.txt("Выбери категорию кнопкой, отметь нужных людей и удали отмеченных", size=13, color=MUTED),
                ft.Container(height=12),
                ft.Row([students_btn, teachers_btn], spacing=10, wrap=True),
                ft.Container(height=14),
                users_list,
                ft.Container(height=14),
                ft.ElevatedButton(
                    "Удалить отмеченных",
                    icon=icon("GROUP_REMOVE_ROUNDED"),
                    bgcolor=RED,
                    color="white",
                    height=48,
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=14)),
                    on_click=lambda e: self.admin_delete_selected_users_confirm(checkbox_login_map, users_list),
                ),
            ], spacing=0),
            expand=True,
        )

        def cleanup_tile(title: str, subtitle: str, icon_name: str, on_click):
            return ft.Container(
                expand=1,
                bgcolor=SURFACE,
                border_radius=18,
                padding=16,
                border=ft.border.all(1, BORDER),
                content=ft.Column([
                    ft.Icon(icon(icon_name), color=AMBER, size=24),
                    ft.Container(height=8),
                    self.txt(title, size=15, weight=ft.FontWeight.W_700),
                    self.txt(subtitle, size=12, color=MUTED),
                    ft.Container(height=12),
                    ft.ElevatedButton(
                        "Очистить",
                        bgcolor=AMBER,
                        color="white",
                        height=42,
                        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)),
                        on_click=lambda e: on_click(),
                    ),
                ], spacing=0),
            )

        cleanup_block = self.card(
            "Очистка данных",
            ft.Column([
                ft.Row([
                    cleanup_tile("Заявки", "Удалить все заявки", "APPROVAL_ROUNDED", self.admin_clear_requests_confirm),
                    cleanup_tile("Журнал", "Очистить все записи", "RECEIPT_LONG_ROUNDED", self.admin_clear_logs_confirm),
                ], spacing=12),
                ft.Container(height=12),
                ft.Row([
                    cleanup_tile("Пропуска", "Удалить временные пропуска", "TIMER_OFF_ROUNDED", self.admin_clear_temp_passes_confirm),
                    cleanup_tile("Фото", "Удалить сохранённые снимки", "PHOTO_LIBRARY_ROUNDED", self.admin_clear_captures_confirm),
                ], spacing=12),
            ], spacing=0),
            expand=True,
        )

        return ft.Column([
            database_list_block,
            ft.Container(height=16),
            user_block,
            ft.Container(height=16),
            cleanup_block,
        ], expand=True, scroll=ft.ScrollMode.AUTO)

    def admin_delete_user(self, login: str) -> None:
        ok, msg = self.store.delete_user(login)
        self.store = DataStore()
        self.snack(msg, ok)
        self.force_refresh_current_view("database")

    def admin_delete_single_user(self, login: Optional[str]) -> None:
        if not login:
            self.snack("Выбери пользователя", False)
            return
        self.admin_delete_user(login)

    def admin_delete_single_user_confirm(self, login: Optional[str]) -> None:
        if not login:
            self.snack("Выбери пользователя", False)
            return
        self.confirm_action(
            "Удалить пользователя?",
            f"Пользователь {login} будет удалён из базы, заявок и временных пропусков.",
            lambda: self.admin_delete_single_user(login),
        )

    def admin_delete_selected_users(self, checkbox_login_map: Dict[int, str], users_list: ft.Column) -> None:
        logins: List[str] = []
        for item in users_list.controls:
            cb = getattr(item, "content", None)
            if cb is not None and getattr(cb, "value", False):
                login = checkbox_login_map.get(id(cb), "")
                if login:
                    logins.append(login)
        ok, msg = self.store.delete_users(logins)
        self.store = DataStore()
        self.snack(msg, ok)
        if ok:
            self.force_refresh_current_view("database")

    def admin_delete_selected_users_confirm(self, checkbox_login_map: Dict[int, str], users_list: ft.Column) -> None:
        logins: List[str] = []
        for item in users_list.controls:
            cb = getattr(item, "content", None)
            if cb is not None and getattr(cb, "value", False):
                login = checkbox_login_map.get(id(cb), "")
                if login:
                    logins.append(login)
        if not logins:
            self.snack("Отметь хотя бы одного пользователя", False)
            return
        self.confirm_action(
            "Удалить отмеченных пользователей?",
            f"Будут удалены: {', '.join(logins)}.",
            lambda: self.admin_delete_selected_users(checkbox_login_map, users_list),
        )

    def admin_delete_all_users(self) -> None:
        ok, msg = self.store.clear_users()
        self.store = DataStore()
        self.snack(msg, ok)
        if ok:
            self.force_refresh_current_view("database")

    def admin_delete_all_users_confirm(self) -> None:
        self.confirm_action(
            "Удалить всех пользователей?",
            "Будут удалены все пользователи, кроме главного admin.",
            self.admin_delete_all_users,
        )

    def admin_clear_logs(self) -> None:
        ok, msg = self.store.clear_logs()
        self.store = DataStore()
        self.snack(msg, ok)
        self.force_refresh_current_view("logs")

    def admin_clear_logs_confirm(self) -> None:
        self.confirm_action(
            "Очистить журнал?",
            "Все записи журнала входов будут удалены.",
            self.admin_clear_logs,
        )

    def admin_clear_requests(self) -> None:
        ok, msg = self.store.clear_requests()
        self.store = DataStore()
        self.snack(msg, ok)
        self.force_refresh_current_view("requests")

    def admin_clear_requests_confirm(self) -> None:
        self.confirm_action(
            "Очистить заявки?",
            "Все заявки на доступ будут удалены.",
            self.admin_clear_requests,
        )

    def admin_clear_temp_passes(self) -> None:
        ok, msg = self.store.clear_temp_passes()
        self.store = DataStore()
        self.snack(msg, ok)
        self.force_refresh_current_view("database")

    def admin_clear_temp_passes_confirm(self) -> None:
        self.confirm_action(
            "Очистить временные пропуска?",
            "Все временные пропуска будут удалены.",
            self.admin_clear_temp_passes,
        )

    def admin_clear_captures(self) -> None:
        ok, msg = self.store.clear_captures()
        self.store = DataStore()
        self.snack(msg, ok)
        self.force_refresh_current_view("database")

    def admin_clear_captures_confirm(self) -> None:
        self.confirm_action(
            "Очистить фото?",
            "Все сохранённые фото и снимки попыток будут удалены.",
            self.admin_clear_captures,
        )

    def admin_requests_page(self) -> ft.Control:
        items = self.store.requests
        header = ft.Row([
            self.txt("Все заявки", size=18, weight=ft.FontWeight.W_700),
            ft.Container(expand=True),
            ft.OutlinedButton(
                "Очистить все",
                icon=icon("DELETE_SWEEP_ROUNDED"),
                on_click=lambda e: self.admin_clear_requests_confirm(),
            ),
        ])
        cards = [self.request_card(x, x.get("status") == "pending") for x in items] or [self.txt("Заявок нет", color=MUTED)]
        return ft.Column([header, ft.Container(height=12)] + cards, spacing=10, expand=True, scroll=ft.ScrollMode.AUTO)

    def admin_logs_page(self) -> ft.Control:
        return self.logs_list(self.store.logs)

    def admin_analytics_page(self) -> ft.Control:
        metrics = self.store.metrics()
        parts = self.store.day_parts()
        max_part = max(parts.values()) if parts else 0
        suspicious_items = [x for x in self.store.logs if x.get("suspicious")][:8]
        zone_map: Dict[str, int] = {}
        role_map: Dict[str, int] = {"student": 0, "teacher": 0, "admin": 0}
        for log in self.store.logs:
            zone = log.get("zone", "—")
            zone_map[zone] = zone_map.get(zone, 0) + 1
        for user in self.store.users:
            role_map[user.get("role", "student")] += 1
        top_zones = sorted(zone_map.items(), key=lambda x: x[1], reverse=True)[:5]

        suspicious_controls = [
            ft.Row([
                self.image_box(item.get("photo_path", ""), 76, 60),
                ft.Column([
                    self.txt(item.get("name", "Неизвестно"), size=13, weight=ft.FontWeight.W_700),
                    self.txt(f"{item.get('time', '')} • {item.get('zone', '')}", size=11, color=MUTED),
                    self.txt(item.get("reason", ""), size=11, color=AMBER),
                ], spacing=2, expand=True),
            ]) for item in suspicious_items
        ] or [self.txt("Подозрительных попыток пока нет", color=MUTED)]

        zone_controls = [
            ft.Row([
                self.txt(zone),
                ft.Container(expand=True),
                self.pill(str(count), PRIMARY_SOFT, PRIMARY),
            ]) for zone, count in top_zones
        ] or [self.txt("Нет данных по зонам", color=MUTED)]

        return ft.Column([
            ft.Row([
                self.metric_card("Разрешено", str(metrics["allowed"]), GREEN, GREEN_SOFT),
                self.metric_card("Отказано", str(metrics["denied"]), RED, RED_SOFT),
                self.metric_card("FAR", f"{metrics['far']}%", BLUE, BLUE_SOFT),
                self.metric_card("FRR", f"{metrics['frr']}%", AMBER, AMBER_SOFT),
            ], spacing=14),
            ft.Container(height=16),
            ft.Row([
                self.card(
                    "График по времени суток",
                    ft.Column([
                        self.progress_row("Утро", parts.get("morning", 0), max_part, BLUE, BLUE_SOFT),
                        self.progress_row("День", parts.get("day", 0), max_part, PRIMARY, PRIMARY_SOFT),
                        self.progress_row("Вечер", parts.get("evening", 0), max_part, PURPLE, PURPLE_SOFT),
                    ], spacing=12),
                    expand=True,
                ),
                self.card(
                    "Пользователи по ролям",
                    ft.Column([
                        self.progress_row("Студенты", role_map["student"], max(role_map.values()) or 1, PRIMARY, PRIMARY_SOFT),
                        self.progress_row("Преподаватели", role_map["teacher"], max(role_map.values()) or 1, GREEN, GREEN_SOFT),
                        self.progress_row("Админы", role_map["admin"], max(role_map.values()) or 1, AMBER, AMBER_SOFT),
                    ], spacing=12),
                    expand=True,
                ),
            ], spacing=16),
            ft.Container(height=16),
            ft.Row([
                self.card("Подозрительные попытки", ft.Column(suspicious_controls, spacing=10), expand=True),
                self.card("Активные зоны", ft.Column(zone_controls, spacing=10), expand=True),
            ], spacing=16),
        ], expand=True, scroll=ft.ScrollMode.AUTO)


def main(page: ft.Page):
    AccessApp(page)

if __name__ == "__main__":
    ft.run(main)