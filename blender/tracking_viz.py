"""
VR tracking visualisation for Blender (tested idea: Blender 4.x / 5.x).

Что делает один запуск (Scripting -> Open -> Alt+P):
  1. парсит лог трекинга (голова, 2 руки по 26 суставов, тело 24 сустава);
  2. строит 3D-сцену: шар головы с "носом", скелеты, пол;
  3. вешает камеру "из глаз" на позу головы;
  4. кладёт видео в Video Sequencer (синхронно с таймлайном);
  5. режет окно на 3 области: вид из камеры | вид сбоку | видео.
Потом достаточно нажать ПРОБЕЛ.

Данные не хранятся в keyframes: обработчик смены кадра берёт значения
из numpy-массивов по реальному времени timeStampNs.
"""
import bpy, json, os, math, glob, traceback
import numpy as np
from mathutils import Quaternion, Vector, Euler

# ============================== CONFIG ==============================
LOG_PATH   = ""      # "" = найти trackingData*.txt рядом со скриптом / .blend / в data/
VIDEO_PATH = ""      # "" = найти mp4/mov/mkv/avi рядом со скриптом / .blend / в data/
FPS        = 30      # должен совпадать с fps видео
RESOLUTION = (1920, 1080)  # разрешение сцены (кадр камеры и итоговый рендер), всегда 16:9
CLEAR_SCENE = True         # удалить из .blend всё, что не создано скриптом (куб, камера, свет)
CAMERA_FOV_DEG = 90
SYNC_SHIFT_S   = -0.25     # сдвиг трекинга относительно видео (сек), если нужно подогнать
SETUP_LAYOUT   = True      # нарезать экран на 3 области
AUTOPLAY       = False     # сразу запустить воспроизведение
SIDE_VIEW_DIST = None      # None = подобрать автоматически (больше = дальше)
# ====================================================================


# ---------------------------- поиск файлов ----------------------------
def candidate_dirs():
    dirs = []
    sd = bpy.context.space_data
    if sd is not None and getattr(sd, "type", "") == 'TEXT_EDITOR' and sd.text and sd.text.filepath:
        dirs.append(os.path.dirname(bpy.path.abspath(sd.text.filepath)))
    for t in bpy.data.texts:
        if t.filepath:
            dirs.append(os.path.dirname(bpy.path.abspath(t.filepath)))
    if bpy.data.filepath:
        dirs.append(os.path.dirname(bpy.data.filepath))
    try:
        dirs.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    dirs.append(os.getcwd())
    out = []
    for d in dirs:
        for c in (d, os.path.join(d, "data"), os.path.dirname(d), os.path.join(os.path.dirname(d), "data")):
            if c not in out and os.path.isdir(c):
                out.append(c)
    return out


VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg")


def find_by_ext(exts):
    """Первый файл с одним из расширений (без учёта регистра) в папках поиска."""
    for d in candidate_dirs():
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for n in names:
            if n.lower().endswith(exts):
                return os.path.join(d, n)
    return None


def clear_scene():
    """Убрать дефолтные куб/камеру/свет (или старые TR_-объекты, если CLEAR_SCENE=False)."""
    for o in list(bpy.data.objects):
        if CLEAR_SCENE or o.name.startswith("TR_"):
            bpy.data.objects.remove(o, do_unlink=True)
    if CLEAR_SCENE:
        try:
            bpy.data.orphans_purge(do_recursive=True)   # заодно удалит меши/свет/камеру-сироты
        except Exception:
            pass


def bind_sequencer_scene(scene):
    """Blender 5.0: Sequencer показывает сцену, привязанную к workspace (кнопка New в его шапке).
    Привязываем основную сцену, чтобы видео шло по тому же таймлайну, что и трекинг."""
    ok = False
    for ws in bpy.data.workspaces:
        if hasattr(ws, "sequencer_scene"):
            try:
                ws.sequencer_scene = scene
                ok = True
            except Exception as e:
                print("не удалось привязать сцену к Sequencer в", ws.name, ":", e)
    return ok


def find_file(patterns):
    for d in candidate_dirs():
        for p in patterns:
            hits = sorted(glob.glob(os.path.join(d, p)))
            if hits:
                return hits[0]
    return None


# ------------------------------ парсинг ------------------------------
def nums(s):
    """'-0,08,-0,02,0,16' -> [-0.08, -0.02, 0.16] (десятичная запятая + запятая-разделитель)"""
    t = s.split(",")
    if len(t) % 2:
        return None
    try:
        return [float(t[i] + "." + t[i + 1]) for i in range(0, len(t), 2)]
    except ValueError:
        return None


def joints_xyz(lst, n):
    out = np.full((n, 3), np.nan)
    for k, j in enumerate(lst[:n]):
        v = nums(j["p"])
        if v and len(v) >= 3:
            out[k] = v[:3]
    return out


def parse(log_path, cache_path):
    T, H, L, R, B = [], [], [], [], []
    video_t = 0
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if isinstance(o, str):
                o = json.loads(o)
            if "notice" in o:                 # служебная строка: время первого кадра видео
                video_t = o["timeStampNs"]
                continue
            T.append(o["timeStampNs"])
            h = nums(o["Head"]["pose"])
            H.append(h if h and len(h) == 7 else [np.nan] * 7)
            hand = o.get("Hand", {})
            for key, dst in (("leftHand", L), ("rightHand", R)):
                hd = hand.get(key)
                if hd and hd.get("isActive"):
                    dst.append(joints_xyz(hd.get("HandJointLocations", []), 26))
                else:
                    dst.append(np.full((26, 3), np.nan))
            B.append(joints_xyz(o.get("Body", {}).get("joints", []), 24))
    np.savez(cache_path, t=np.array(T, dtype=np.int64), head=np.array(H), L=np.array(L),
             R=np.array(R), B=np.array(B), video_t=np.int64(video_t),
             src_size=np.int64(os.path.getsize(log_path)))


def load_data(log_path):
    if log_path.lower().endswith(".npz"):      # уже готовый кэш, парсить нечего
        return np.load(log_path)
    cache = log_path + ".cache.npz"
    ok = False
    if os.path.exists(cache):
        try:
            ok = int(np.load(cache)["src_size"]) == os.path.getsize(log_path)
        except Exception:
            ok = False
    if not ok:
        print("Парсинг лога (первый запуск, может занять до минуты)...")
        parse(log_path, cache)
    return np.load(cache)


# ------------------------- сцена и геометрия -------------------------
def new_collection(name):
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if col.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(col)
    return col


def material(name, color):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = color
    return m


def unit_sphere_mesh():
    bpy.ops.mesh.primitive_uv_sphere_add(segments=10, ring_count=6, radius=1)
    o = bpy.context.active_object
    me = o.data
    bpy.data.objects.remove(o, do_unlink=True)
    return me


def make_spheres(col, mesh, prefix, n, radius, color):
    m = material("TR_mat_" + prefix, color)
    objs = []
    for i in range(n):
        o = bpy.data.objects.new(f"TR_{prefix}_{i}", mesh)
        o.scale = (radius,) * 3
        o.data.materials.clear()
        o.data.materials.append(m)
        col.objects.link(o)
        objs.append(o)
    return objs


def make_bones(col, name, edges, radius, color):
    """Кости как тонкие 4-гранные призмы: видны и в обычном рендере. Возвращает функцию update(pts)."""
    n = len(edges)
    faces = []
    for k in range(n):
        o = 8 * k
        for j in range(4):
            j2 = (j + 1) % 4
            faces.append((o + j, o + j2, o + 4 + j2, o + 4 + j))
    me = bpy.data.meshes.new("TR_" + name)
    me.from_pydata([(0, 0, 0)] * (8 * n), [], faces)
    me.materials.append(material("TR_mat_" + name, color))
    obj = bpy.data.objects.new("TR_" + name, me)
    col.objects.link(obj)
    ea = np.array([e[0] for e in edges])
    eb = np.array([e[1] for e in edges])

    def update(pts):
        a, b = pts[ea], pts[eb]
        ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        a = np.where(ok[:, None], a, 0.0)
        b = np.where(ok[:, None], b, 0.0)
        d = b - a
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        helper = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
        u = np.cross(d, helper)
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        v = np.cross(d, u)
        r = radius * ok[:, None]
        ring = np.stack([u, v, -u, -v], axis=1) * r[:, None, :]
        co = np.concatenate([a[:, None, :] + ring, b[:, None, :] + ring], axis=1).astype(np.float32)
        me.vertices.foreach_set("co", co.ravel())
        me.update()

    return update


def add_op_object(col, name, op, **kw):
    op(**kw)
    o = bpy.context.active_object
    o.name = name
    for c in list(o.users_collection):
        c.objects.unlink(o)
    col.objects.link(o)
    return o


# ------------------------------ раскладка ------------------------------
def split(win, area, direction, factor=0.5):
    before = {a.as_pointer() for a in win.screen.areas}
    with bpy.context.temp_override(window=win, screen=win.screen, area=area):
        bpy.ops.screen.area_split(direction=direction, factor=factor)
    return [a for a in win.screen.areas if a.as_pointer() not in before][0]


def clean_view(space):
    space.shading.type = 'SOLID'
    space.overlay.show_overlays = False
    space.show_gizmo = False
    space.show_region_ui = False
    space.show_region_toolbar = False
    space.show_region_header = False


def build_layout(side_target, side_dist, has_video):
    try:
        win = bpy.context.window_manager.windows[0]
        sc = win.scene
        areas3d = [a for a in win.screen.areas if a.type == 'VIEW_3D']
        main = max(areas3d, key=lambda a: a.width * a.height)

        new = split(win, main, 'VERTICAL')
        left, right = sorted([main, new], key=lambda a: a.x)
        new2 = split(win, right, 'HORIZONTAL')
        top, bottom = sorted([right, new2], key=lambda a: -a.y)

        # слева: вид из камеры головы
        cam = bpy.data.objects.get("TR_Camera")
        if cam:
            sc.camera = cam
        sp = left.spaces.active
        clean_view(sp)
        sp.region_3d.view_perspective = 'CAMERA'

        # справа сверху: вид сбоку (как Numpad 3)
        sp = top.spaces.active
        clean_view(sp)
        r3d = sp.region_3d
        r3d.view_perspective = 'ORTHO'
        r3d.view_rotation = Euler((math.radians(90), 0, math.radians(90))).to_quaternion()
        r3d.view_location = side_target
        r3d.view_distance = side_dist

        # справа снизу: видео
        if has_video:
            bottom.type = 'SEQUENCE_EDITOR'
            bottom.spaces.active.view_type = 'PREVIEW'

        for a in win.screen.areas:
            a.tag_redraw()
        sc.frame_set(sc.frame_start)

        if AUTOPLAY:
            with bpy.context.temp_override(window=win, screen=win.screen, area=left):
                bpy.ops.screen.animation_play()
        print("Раскладка готова. Нажмите ПРОБЕЛ для воспроизведения.")
    except Exception:
        traceback.print_exc()
    return None


# ------------------------------- основное -------------------------------
def main():
    log_path = LOG_PATH or find_file(["trackingData*.txt", "*tracking*.txt", "*.npz"])
    if not log_path or not os.path.exists(log_path):
        raise RuntimeError("Не найден лог трекинга. Положите trackingData_*.txt (или готовый .npz) "
                           "рядом со скриптом или укажите LOG_PATH в блоке CONFIG.")
    video_path = VIDEO_PATH or find_by_ext(VIDEO_EXT)

    d = load_data(log_path)
    t_ns, HEAD, LH, RH, BODY = d["t"], d["head"], d["L"], d["R"], d["B"]
    video_t = int(d["video_t"])
    t_s = (t_ns - t_ns[0]) / 1e9
    N = len(t_s)

    # --- конвенция осей: считаем, по какую сторону от взгляда головы лежат руки ---
    score = []
    for i in range(0, N, 50):
        h, w = HEAD[i], LH[i][1]
        if np.isnan(h).any() or np.isnan(w).any():
            continue
        q = Quaternion((h[6], h[3], h[4], h[5]))
        fwd = q @ Vector((0, 0, 1))
        score.append(fwd.dot(Vector(w) - Vector(h[:3])))
    fwd_score = float(np.mean(score)) if score else 0.0
    UNITY = fwd_score > 0   # руки впереди по +Z -> Unity-конвенция

    def conv_pos(a):
        a = np.asarray(a)
        x, y, z = a[..., 0], a[..., 1], a[..., 2]
        return np.stack([x, z if UNITY else -z, y], axis=-1)

    def conv_quat(x, y, z, w):   # -> Blender (w, x, y, z)
        return (w, -x, -z, -y) if UNITY else (w, x, -z, y)

    LHb, RHb, BODYb, HEADb = conv_pos(LH), conv_pos(RH), conv_pos(BODY), conv_pos(HEAD[:, :3])

    # --- пол на Z=0 ---
    floor_z = np.nanpercentile(BODYb[..., 2], 1)
    for arr in (LHb, RHb, BODYb, HEADb):
        arr[..., 2] -= floor_z

    # --- центр и размах для пола и бокового вида ---
    pelvis = BODYb[:, 0, :]
    cx, cy = float(np.nanmedian(pelvis[:, 0])), float(np.nanmedian(pelvis[:, 1]))
    extent = float(max(np.nanmax(pelvis[:, 0]) - np.nanmin(pelvis[:, 0]),
                       np.nanmax(pelvis[:, 1]) - np.nanmin(pelvis[:, 1])))
    side_target = (cx, cy, 0.9)
    side_dist = SIDE_VIEW_DIST or max(4.0, extent * 2.0 + 2.0)

    # --- чистим сцену: дефолтные куб/камера/свет и старые TR_-объекты ---
    clear_scene()
    col = new_collection("Tracking")
    sphere = unit_sphere_mesh()

    # --- объекты ---
    HAND_EDGES = []
    for chain in ([1, 2, 3, 4, 5], [1, 6, 7, 8, 9, 10], [1, 11, 12, 13, 14, 15],
                  [1, 16, 17, 18, 19, 20], [1, 21, 22, 23, 24, 25]):
        HAND_EDGES += list(zip(chain[:-1], chain[1:]))
    PARENT = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9,
              13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}
    BODY_EDGES = list(PARENT.items())

    cL, cR, cB = (0.2, 0.6, 1, 1), (1, 0.3, 0.3, 1), (0.9, 0.9, 0.2, 1)
    left_o = make_spheres(col, sphere, "L", 26, 0.008, cL)
    right_o = make_spheres(col, sphere, "R", 26, 0.008, cR)
    body_o = make_spheres(col, sphere, "B", 24, 0.025, cB)
    left_b = make_bones(col, "Lbones", HAND_EDGES, 0.003, cL)
    right_b = make_bones(col, "Rbones", HAND_EDGES, 0.003, cR)
    body_b = make_bones(col, "Bbones", BODY_EDGES, 0.012, cB)

    head_o = bpy.data.objects.new("TR_Head", sphere)
    head_o.scale = (0.09, 0.09, 0.09)
    head_o.rotation_mode = 'QUATERNION'
    head_o.data.materials.append(material("TR_mat_head", (0.8, 0.8, 0.85, 1)))
    col.objects.link(head_o)
    nose = add_op_object(col, "TR_Nose", bpy.ops.mesh.primitive_cone_add, radius1=0.5, depth=1.5)
    nose.rotation_euler = (-math.pi / 2, 0, 0)
    nose.scale = (0.5, 0.5, 0.5)
    nose.location = (0, 1.0, 0)
    nose.data.materials.append(material("TR_mat_nose", (1, 0.5, 0.1, 1)))
    nose.parent = head_o

    floor_size = max(6.0, extent + 4.0)
    floor = add_op_object(col, "TR_Floor", bpy.ops.mesh.primitive_plane_add,
                          size=floor_size, location=(cx, cy, 0))
    floor.data.materials.append(material("TR_mat_floor", (0.22, 0.22, 0.26, 1)))

    # --- камера из глаз (пустышка без масштаба + камера) ---
    rig = bpy.data.objects.new("TR_CamRig", None)
    rig.rotation_mode = 'QUATERNION'
    col.objects.link(rig)
    cam_data = bpy.data.cameras.new("TR_CamData")
    cam_data.angle = math.radians(CAMERA_FOV_DEG)
    cam_data.clip_start = 0.15
    cam = bpy.data.objects.new("TR_Camera", cam_data)
    col.objects.link(cam)
    cam.parent = rig
    cam.rotation_euler = (math.pi / 2, 0, 0)

    sc = bpy.context.scene
    sc.camera = cam

    # --- обновление по кадру ---
    def set_spheres(objs, pts):
        for o, p in zip(objs, pts):
            bad = bool(np.isnan(p).any())
            o.hide_viewport = bad
            o.hide_render = bad
            if not bad:
                o.location = p

    def tracking_update(scene, *args):
        t = (scene.frame_current - scene.frame_start) / FPS + SYNC_SHIFT_S
        i = int(np.clip(np.searchsorted(t_s, t), 0, N - 1))
        set_spheres(left_o, LHb[i]);   left_b(LHb[i])
        set_spheres(right_o, RHb[i]);  right_b(RHb[i])
        set_spheres(body_o, BODYb[i]); body_b(BODYb[i])
        h = HEAD[i]
        if not np.isnan(h).any():
            q = conv_quat(*h[3:7])
            head_o.location = HEADb[i]
            head_o.rotation_quaternion = q
            rig.location = HEADb[i]
            rig.rotation_quaternion = q

    hl = bpy.app.handlers.frame_change_post
    for fn in list(hl):
        if fn.__name__ == "tracking_update":
            hl.remove(fn)
    hl.append(tracking_update)

    sc.render.fps = FPS
    sc.render.resolution_x, sc.render.resolution_y = RESOLUTION
    sc.render.resolution_percentage = 100
    sc.frame_start = 1
    sc.frame_end = int(t_s[-1] * FPS) + 1

    # --- видео в Sequencer ---
    has_video = False
    seq_bound = None
    video_note = ("видео не найдено. Искал " + "/".join(VIDEO_EXT) + " в: " + "; ".join(candidate_dirs())
                  + ". Положите видео в одну из этих папок или задайте VIDEO_PATH.")
    if VIDEO_PATH and not os.path.exists(VIDEO_PATH):
        video_note = f"VIDEO_PATH указан, но файла нет: {VIDEO_PATH}"
    if video_path and os.path.exists(video_path):
        if sc.sequence_editor is None:
            sc.sequence_editor_create()
        se = sc.sequence_editor
        strips = se.strips if hasattr(se, "strips") else se.sequences
        for s in list(strips):
            if s.name in ("TR_Video", "TR_Audio"):
                strips.remove(s)
        st = strips.new_movie("TR_Video", video_path, channel=1, frame_start=1)
        try:
            strips.new_sound("TR_Audio", video_path, channel=2, frame_start=1)
        except Exception as e:
            print("звук не добавлен:", e)
        has_video = True
        seq_bound = bind_sequencer_scene(sc)
        warn = ""
        if abs(st.fps - FPS) > 0.5:
            warn = f"  ВНИМАНИЕ: fps видео {st.fps:.2f} != FPS={FPS}, поменяйте FPS в CONFIG"
        video_note = f"{os.path.basename(video_path)}, fps видео {st.fps:.2f}{warn}"

    sc.frame_set(1)

    # --- отчёт ---
    rep = [
        f"лог: {log_path}",
        f"оси: {'Unity (+Z вперёд)' if UNITY else 'OpenXR (-Z вперёд)'} (оценка {fwd_score:.3f})",
        f"записей трекинга: {N}, длительность {t_s[-1]:.1f} с, ~{N / t_s[-1]:.1f} Гц",
        f"кадров анимации: {sc.frame_end} при {FPS} fps",
        f"кадр видео - начало трекинга: {(video_t - t_ns[0]) / 1e9:.3f} с",
        f"правая рука без данных: {int(np.isnan(RH[:, 0, 0]).sum())} записей",
        f"видео: {video_note}",
        "сцена привязана к Sequencer: " + ("да" if seq_bound else "нет (в этой версии Blender не требуется или видео нет)"),
    ]
    txt = bpy.data.texts.get("report") or bpy.data.texts.new("report")
    txt.clear()
    txt.write("\n".join(rep))
    print("\n".join(rep))

    # --- раскладка экрана ---
    if SETUP_LAYOUT:
        win = bpy.context.window_manager.windows[0]
        ws = bpy.data.workspaces.get("Layout")
        if ws:
            win.workspace = ws
        bpy.app.timers.register(lambda: build_layout(side_target, side_dist, has_video),
                                first_interval=0.4)


try:
    main()
except Exception:
    err = traceback.format_exc()
    print(err)
    t = bpy.data.texts.get("error") or bpy.data.texts.new("error")
    t.clear()
    t.write(err)
    raise
