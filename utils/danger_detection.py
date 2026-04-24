# coding: utf-8
"""
危险人员检测模块 —— 高速移动防抖增强版（已修复报错）
"""
import numpy as np
import cv2

class PersonTracker:
    """
    人员状态跟踪器
    用于跟踪人员的危险状态，实现连续一段时间不危险才判定为不危险的机制
    同时实现连续一段时间危险才判定为危险的机制，避免误检
    """

    def __init__(self):
        self.person_states = {}
        self.safe_frames_threshold = 12
        self.danger_frames_threshold = 2
        self.person_id_counter = 0
        self.person_history = {}

        # 武器记忆，解决高速移动丢帧
        self.weapon_memory = {}
        self.max_weapon_age = 6

    def get_person_id(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        best_match_id = None
        best_match_score = 0.5

        for person_id, history in self.person_history.items():
            hx1, hy1, hx2, hy2 = history['bbox']
            hcx = (hx1 + hx2) / 2
            hcy = (hy1 + hy2) / 2
            dist = np.hypot(center_x - hcx, center_y - hcy)
            s1 = (x2 - x1) * (y2 - y1)
            s2 = (hx2 - hx1) * (hy2 - hy1)
            ratio = min(s1, s2) / max(s1, s2)
            sim = ratio * (1 - dist / max(x2 - x1, y2 - y1))
            if sim > best_match_score:
                best_match_score = sim
                best_match_id = person_id

        if best_match_id is not None:
            self.person_history[best_match_id]['bbox'] = bbox
            return best_match_id
        else:
            new_id = f"person_{self.person_id_counter}"
            self.person_id_counter += 1
            self.person_history[new_id] = {'bbox': bbox}
            return new_id

    def update_person_state(self, persons, weapons):
        # -------------------- 武器缓存 --------------------
        current_keys = []
        for w in weapons:
            cx = (w['bbox'][0] + w['bbox'][2]) / 2
            cy = (w['bbox'][1] + w['bbox'][3]) / 2
            key = f"{cx:.1f}_{cy:.1f}"
            self.weapon_memory[key] = {"weapon": w, "age": 0}
            current_keys.append(key)

        expired = []
        for k in self.weapon_memory:
            if k not in current_keys:
                self.weapon_memory[k]["age"] += 1
                if self.weapon_memory[k]["age"] > self.max_weapon_age:
                    expired.append(k)
        for k in expired:
            del self.weapon_memory[k]

        # 合并当前+历史武器
        all_weapons = weapons + [v["weapon"] for v in self.weapon_memory.values() if v["age"] > 0]

        # -------------------- 危险判定 --------------------
        current_danger = {}
        for p in persons:
            pid = self.get_person_id(p['bbox'])
            px1, py1, px2, py2 = p['bbox']
            danger = False

            for w in all_weapons:
                wx1, wy1, wx2, wy2 = w['bbox']
                ix1 = max(px1, wx1)
                iy1 = max(py1, wy1)
                ix2 = min(px2, wx2)
                iy2 = min(py2, wy2)
                iarea = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                warea = (wx2 - wx1) * (wy2 - wy1) + 1e-6

                if iarea / warea > 0.2:
                    danger = True
                    break
                wcx = (wx1 + wx2) / 2
                wcy = (wy1 + wy2) / 2
                if px1 < wcx < px2 and py1 < wcy < py2:
                    danger = True
                    break

            current_danger[pid] = danger

        # -------------------- 状态机（已修复报错） --------------------
        for p in persons:
            pid = self.get_person_id(p['bbox'])
            if pid not in self.person_states:
                self.person_states[pid] = {
                    'is_danger': False,
                    'safe_frames': 0,
                    'danger_frames': 0
                }

            state = self.person_states[pid]

            if current_danger[pid]:
                state['danger_frames'] += 1
                state['safe_frames'] = 0
                if state['danger_frames'] >= self.danger_frames_threshold:
                    state['is_danger'] = True
            else:
                state['safe_frames'] += 1
                state['danger_frames'] = 0
                if state['safe_frames'] >= self.safe_frames_threshold:
                    state['is_danger'] = False

            p['is_danger'] = state['is_danger']

        return persons


def detect_danger_persons(persons, weapons):
    for person in persons:
        person['is_danger'] = False
        px1, py1, px2, py2 = person['bbox']
        for weapon in weapons:
            wx1, wy1, wx2, wy2 = weapon['bbox']
            ix1 = max(px1, wx1)
            iy1 = max(py1, wy1)
            ix2 = min(px2, wx2)
            iy2 = min(py2, wy2)
            iarea = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            warea = (wx2 - wx1) * (wy2 - wy1) + 1e-6
            if iarea / warea > 0.2:
                person['is_danger'] = True
                break
            wcx = (wx1 + wx2) / 2
            wcy = (wy1 + wy2) / 2
            if px1 < wcx < px2 and py1 < wcy < py2:
                person['is_danger'] = True
                break
    return persons


def draw_detection_results(image, persons, weapons, weapon_names):
    for person in persons:
        x1, y1, x2, y2 = person['bbox']
        if person['is_danger']:
            color = (0, 0, 255)
            label = 'Danger Person'
        else:
            color = (0, 255, 0)
            label = 'Person'
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for weapon in weapons:
        x1, y1, x2, y2 = weapon['bbox']
        cls = weapon['class']
        conf = weapon['conf']
        color = (255, 0, 0)
        label = f'{weapon_names[cls]} {conf:.2f}'
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image