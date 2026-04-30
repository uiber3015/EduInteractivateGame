import json
import os
import re
from typing import Dict, List, Tuple

STYLE_BIBLE_PROMPT = "Warm hand-painted Ghibli-like educational science-adventure illustration. Soft cinematic lighting, gentle storybook atmosphere, anime-style character design, painterly textures, and a polished non-photorealistic finish."
NON_PHOTOREALISTIC_RULE = "Do not render as realistic photography, studio portrait photography, live-action film still, glossy advertising image, or hyperreal commercial product shot. Maintain a warm hand-painted illustrated look throughout."
NO_TEXT_IMAGE_RULE = "画面中绝对不要出现任何可读文字、中文、英文、数字、数学符号、公式符号、标点符号、logo、水印、标签、海报文字、屏幕文字、路牌文字、书本文字、按钮文案或UI文案；若场景包含牌子、屏幕、书本、展板、仪器刻度或包装，只保留其外形，不显示任何可识别内容。"
ANTI_ARTIFACT_RULE = "Do not generate any letters, words, numbers, symbols, glyph-like marks, fake subtitles, watermark-like textures, poster-like patterns, UI fragments, scribbles, formula fragments, punctuation marks, or blurry text-shaped artifacts anywhere in the image. Keep walls, glass, boards, labels, screens, packaging, instruments, and background surfaces visually clean and free of pseudo-text noise."
FINAL_IMAGE_QUALITY_RULE = "The output must be a fully finished, beautiful, production-ready illustration rather than a rough draft. Keep the whole frame high-definition, clean, refined, complete, and visually polished in a 16:9 cinematic composition. Do not introduce noise, dirty textures, heavy oil-paint brush mess, blur, pixel breakup, compression-like artifacts, broken brush fragments, contaminated patches, ghosting, smeared details, cropped-off subjects, or any distracting visual pollution anywhere in the frame. All visible areas must look intentional, detailed, polished, interference-free, and suitable for direct final use."
DEFAULT_PROTAGONIST_ID = "student_protagonist_main"
MANUAL_GENERATION_PREFACE = "这是基于当前场景重新生成的新图像，不是对参考图进行融图或拼接；参考图仅用于保持主角角色形象一致，角色的动作、神态、姿态和构图必须符合当前场景描述。"

MANUAL_ROLE_RULES = {
    DEFAULT_PROTAGONIST_ID: "主角外貌、发型、年龄感、服装和整体轮廓需跨场景保持一致。",
    "peer_students_generic": "若出现其他学生，仅作次要陪衬，不可替代主角成为主体。",
    "teacher_or_staff_generic": "若出现老师或工作人员，仅作次要角色，不可压过主角。",
}

OBJECT_RULES = [
    ("铁球", "rusted_iron_ball", "A solid iron ball whose material, size, and surface state must stay physically consistent across scenes."),
    ("乒乓球", "ping_pong_ball", "A standard white ping-pong ball with stable size, material, and flight behavior."),
    ("吹风机", "hair_dryer", "A handheld hair dryer with stable shape, nozzle direction, and grip orientation."),
    ("纸杯电话", "paper_cup_phone", "A paper-cup phone made of two paper cups connected by a cotton string; keep its structure consistent."),
    ("棉绳", "cotton_string", "A cotton string whose tension, wetness, thickness, and straightness must match the described state."),
    ("轨道", "loop_track", "A smooth metal loop track with stable geometry, slope, and ring size."),
    ("线圈", "coil_device", "A visible coil used in electromagnetic induction; keep its shape and relative position to the magnets stable."),
    ("磁铁", "magnet_pair", "A large magnet pair used in the induction device; keep polarity-facing layout and scale stable."),
    ("检流计", "galvanometer", "A galvanometer meter connected to the induction device; keep its appearance and position stable."),
    ("集章卡", "stamp_card", "A golden science fair stamp card with stable color, size, and handheld usage."),
    ("夹具", "long_handle_clamp", "A long-handled clamp used to retrieve objects from water; keep length and shape stable."),
    ("水槽", "transparent_water_tank", "A large transparent water tank with stable depth and placement."),
]


def _normalize_reference_config(data: Dict) -> Dict[str, str]:
    normalized = {}
    if not isinstance(data, dict):
        return normalized
    for role_id, value in data.items():
        if isinstance(value, str):
            normalized[role_id] = value
        elif isinstance(value, dict):
            ref_url = value.get("url") or value.get("reference_url") or value.get("local_path") or ""
            if ref_url:
                normalized[role_id] = ref_url
    return normalized


def load_character_reference_urls(output_dir: str) -> Dict[str, str]:
    candidates = [
        os.path.join(output_dir, "character_reference_urls.json"),
        os.path.join(output_dir, "output", "character_reference_urls.json"),
        os.path.join(output_dir, "output", "image_consistency", "character_reference_urls.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _normalize_reference_config(data)
    return {}


def _detect_objects(text: str) -> List[Dict[str, str]]:
    found = []
    for keyword, object_id, description in OBJECT_RULES:
        if keyword in text:
            found.append({
                "object_id": object_id,
                "keyword": keyword,
                "description": description,
            })
    return found


def _sanitize_base_prompt(prompt: str) -> str:
    sanitized = (prompt or "").strip()
    sanitized = re.sub(r"[，。,.、；;]?人物形象参考输入的第一张[，。,.、；;]?", "", sanitized)
    sanitized = re.sub(r"[，。,.、；;]?背景参考第二张（如果有的话）[，。,.、；;]?", "", sanitized)
    sanitized = re.sub(r"[，。,.、；;]?背景参考第二张\(如果有的话\)[，。,.、；;]?", "", sanitized)
    sanitized = re.sub(r"[，。,.、；;]?参考输入的第一张[，。,.、；;]?", "", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = sanitized.rstrip("，。,.、；; ")
    return sanitized


def _infer_default_roles(node: dict) -> List[str]:
    content = node.get("content", "") or ""
    metadata = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
    roles = []

    if any(token in content for token in ["你", "你站在", "你来到", "你选择", "你将", "你紧握", "你示意"]):
        roles.append(DEFAULT_PROTAGONIST_ID)

    if any(token in content for token in ["同学", "两名同学", "围观同学", "挑战者"]):
        roles.append("peer_students_generic")

    if any(token in content for token in ["老师", "工作人员"]):
        roles.append("teacher_or_staff_generic")

    recurring = metadata.get("recurring_characters", [])
    if isinstance(recurring, list):
        for item in recurring:
            if item and item not in roles:
                roles.append(item)

    return roles


def build_image_consistency_plan(story_graph: dict, character_reference_urls: Dict[str, str] = None) -> Dict:
    character_reference_urls = character_reference_urls or {}
    plan = {
        "style_bible": {
            "visual_style": "Warm hand-painted Ghibli-like educational science-adventure illustration",
            "global_rules": [
                "Keep the protagonist identity stable across all scenes.",
                "Keep recurring props and devices visually stable across scenes.",
                "Respect the physical setup and state described by the story node.",
                "Do not add readable text into the image.",
            ],
        },
        "characters": [
            {
                "character_id": DEFAULT_PROTAGONIST_ID,
                "display_name": "Student protagonist",
                "required": True,
                "reference_url": character_reference_urls.get(DEFAULT_PROTAGONIST_ID, ""),
                "description": "A middle-school student protagonist in a science fair adventure. Keep age, face, hairstyle, outfit, and overall silhouette stable across all chapters.",
            }
        ],
        "nodes": [],
    }

    optional_roles = {
        "peer_students_generic": "Background or supporting students appearing in some chapters.",
        "teacher_or_staff_generic": "Teacher or staff members who may appear as supporting figures.",
    }

    for role_id, desc in optional_roles.items():
        if character_reference_urls.get(role_id):
            plan["characters"].append({
                "character_id": role_id,
                "display_name": role_id,
                "required": False,
                "reference_url": character_reference_urls.get(role_id, ""),
                "description": desc,
            })

    for node in story_graph.get("nodes", []):
        content = node.get("content", "") or ""
        node_roles = _infer_default_roles(node)
        node_objects = _detect_objects(content + "\n" + json.dumps(node.get("metadata", {}), ensure_ascii=False))
        plan["nodes"].append({
            "node_id": node.get("id"),
            "roles": node_roles,
            "objects": node_objects,
        })

    return plan


def _build_role_constraints(role_ids: List[str], character_reference_urls: Dict[str, str]) -> Tuple[List[str], List[str]]:
    lines = []
    ref_urls = []
    has_protagonist = DEFAULT_PROTAGONIST_ID in role_ids
    only_protagonist = bool(role_ids) and set(role_ids) == {DEFAULT_PROTAGONIST_ID}

    for role_id in role_ids:
        if role_id == DEFAULT_PROTAGONIST_ID:
            lines.append("Use the protagonist as the same canonical student character across all scenes. Keep face shape, hairstyle, age, outfit, and body proportions consistent.")
            lines.append("The protagonist must remain the primary and unmistakable subject of the image, matching the provided reference image rather than a newly invented student.")
            lines.append("Do not change the protagonist into a different boy, girl, classmate, or alternate anime character. Do not swap gender presentation, facial identity, hairstyle, clothing, or silhouette.")
        elif role_id == "peer_students_generic":
            lines.append("If supporting students appear, keep them secondary and visually generic rather than introducing a new dominant character.")
        elif role_id == "teacher_or_staff_generic":
            lines.append("If a teacher or staff member appears, keep them as a secondary supporting figure and do not let them replace the protagonist as the visual focus.")
        else:
            lines.append(f"Keep the recurring role '{role_id}' visually stable if present.")

        ref_url = character_reference_urls.get(role_id)
        if ref_url and ref_url not in ref_urls:
            ref_urls.append(ref_url)

    if only_protagonist:
        lines.append("Show only one clear student protagonist in the foreground. Do not add another foreground student, especially not a female-student replacement or a second competing main character.")
        lines.append("If any extra people are unavoidable, keep them tiny, blurred, background-only extras and never let them become the subject near the experiment setup.")
    elif has_protagonist:
        lines.append("Even if other people appear, the protagonist must stay closest to camera or most visually dominant, and no supporting character may replace the protagonist.")

    return lines, ref_urls


def _build_object_constraints(objects: List[Dict[str, str]]) -> List[str]:
    lines = []
    for item in objects:
        lines.append(item["description"])
    return lines


def _build_manual_role_constraints(role_ids: List[str]) -> List[str]:
    lines = []
    seen = set()
    for role_id in role_ids:
        line = MANUAL_ROLE_RULES.get(role_id)
        if line and line not in seen:
            lines.append(line)
            seen.add(line)
    if DEFAULT_PROTAGONIST_ID not in role_ids:
        fallback = MANUAL_ROLE_RULES.get(DEFAULT_PROTAGONIST_ID)
        if fallback and fallback not in seen:
            lines.append(fallback)
    return lines


def _build_style_constraints() -> List[str]:
    return [
        STYLE_BIBLE_PROMPT,
        "Keep the same protagonist identity, scene storytelling warmth, and illustrated visual language across all chapters.",
        "Use a gentle hand-painted anime storybook look with warm educational-adventure mood and soft cinematic light.",
        "Keep the image in a clean, high-definition, delicate 16:9 composition with complete framing and polished detail.",
        NON_PHOTOREALISTIC_RULE,
    ]


def _build_manual_object_constraints(objects: List[Dict[str, str]], limit: int = 3) -> List[str]:
    if not objects:
        return []
    keywords = []
    seen = set()
    for item in objects:
        keyword = item.get("keyword")
        if keyword and keyword not in seen:
            keywords.append(keyword)
            seen.add(keyword)
        if len(keywords) >= limit:
            break
    if not keywords:
        return []
    examples = "、".join(keywords)
    return [
        "本场景涉及到的关键道具、装置或实验物体，需与前文保持外形、尺寸、材质和状态一致。",
        f"如{examples}等元素出现时，需延续前文设定，不要随意改形或替换。",
    ]


def apply_image_consistency_enrichment(story_graph: dict, character_reference_urls: Dict[str, str] = None) -> Tuple[dict, Dict]:
    character_reference_urls = character_reference_urls or {}
    plan = build_image_consistency_plan(story_graph, character_reference_urls)
    plan_by_node = {item["node_id"]: item for item in plan.get("nodes", [])}

    for node in story_graph.get("nodes", []):
        metadata = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
        if not isinstance(node.get("metadata"), dict):
            node["metadata"] = metadata

        node_plan = plan_by_node.get(node.get("id"), {})
        role_ids = node_plan.get("roles", [])
        objects = node_plan.get("objects", [])

        role_constraints, role_reference_urls = _build_role_constraints(role_ids, character_reference_urls)
        object_constraints = _build_object_constraints(objects)

        manual_override = (metadata.get("manual_image_prompt_override") or "").strip()
        if manual_override:
            protagonist_ref = character_reference_urls.get(DEFAULT_PROTAGONIST_ID)
            if protagonist_ref and protagonist_ref not in role_reference_urls:
                role_reference_urls = [protagonist_ref, *role_reference_urls]
            manual_role_constraints = _build_manual_role_constraints(role_ids)
            manual_object_constraints = _build_manual_object_constraints(objects)
            style_constraints = _build_style_constraints()
            manual_sections = [MANUAL_GENERATION_PREFACE]
            manual_sections.extend(style_constraints)
            manual_sections.append(manual_override)
            manual_sections.extend(role_constraints)
            manual_sections.extend(manual_role_constraints)
            manual_sections.extend(manual_object_constraints)
            manual_sections.append(FINAL_IMAGE_QUALITY_RULE)
            manual_sections.append(ANTI_ARTIFACT_RULE)
            manual_sections.append(NO_TEXT_IMAGE_RULE)
            metadata["consistency_roles"] = role_ids
            metadata["consistency_objects"] = objects
            metadata["role_reference_urls"] = role_reference_urls
            metadata["composed_image_prompt"] = "\n".join([line for line in manual_sections if line])
            continue

        base_prompt = metadata.get("image_prompt") or node.get("content") or ""
        base_prompt = _sanitize_base_prompt(base_prompt)
        if not base_prompt:
            continue

        prompt_sections = [
            base_prompt.strip(),
            STYLE_BIBLE_PROMPT,
            "[Consistency Requirements]",
            "Keep the same visual style, protagonist identity, and object continuity as earlier scenes.",
        ]
        prompt_sections.extend(role_constraints)
        prompt_sections.extend(object_constraints)

        if metadata.get("chapter_title"):
            prompt_sections.append(f"Chapter context: {metadata.get('chapter_title')}.")
        if metadata.get("knowledge_point"):
            prompt_sections.append(f"The image should stay grounded in the scientific setup of: {metadata.get('knowledge_point')}.")

        prompt_sections.append(FINAL_IMAGE_QUALITY_RULE)
        prompt_sections.append(NON_PHOTOREALISTIC_RULE)
        prompt_sections.append(ANTI_ARTIFACT_RULE)
        prompt_sections.append(NO_TEXT_IMAGE_RULE)

        metadata["consistency_roles"] = role_ids
        metadata["consistency_objects"] = objects
        metadata["role_reference_urls"] = role_reference_urls
        metadata["composed_image_prompt"] = "\n".join([line for line in prompt_sections if line])

    return story_graph, plan


def save_image_consistency_artifacts(output_dir: str, story_graph: dict, plan: Dict, artifacts_dir: str = None) -> Tuple[str, str]:
    artifacts_dir = artifacts_dir or os.path.join(output_dir, "output", "image_consistency")
    os.makedirs(artifacts_dir, exist_ok=True)

    plan_path = os.path.join(artifacts_dir, "image_consistency_plan.json")
    graph_path = os.path.join(artifacts_dir, "story_graph_with_consistency_prompts.json")

    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(story_graph, f, ensure_ascii=False, indent=2)

    return plan_path, graph_path
