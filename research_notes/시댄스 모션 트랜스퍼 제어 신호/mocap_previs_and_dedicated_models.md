# Route A (video → 3D mocap previs) and Route B (dedicated motion-transfer models) for live-action → 2D character conversion — state as of November 2026

Research date: 2026-11-15. Source-quality note: many 2026 pages on this topic are vendor or SEO blogs. Primary sources (repos, licence files, official guides, arXiv) are marked **[P]**; secondary/vendor-adjacent pages are marked **[S]**. Where a claim came only from a search-result summary and the page itself could not be opened, it is marked **[snippet]**.

---

## A1. Which markerless video mocap tools/models are current (open research code and commercial products)?

### Takeaway
The best open models (GVHMR, PromptHMR, and anything that outputs SMPL/SMPL-X) are licensed for non-commercial use only, so they are not a clean option for a commercial studio; the realistic commercial choices are cloud services (Autodesk Flow Studio, Move.ai, QuickMagic, DeepMotion, Plask, Rokoko Vision) plus Meta's SAM 3D Body (commercially usable licence, but per-image, not a finished video-mocap pipeline). Meshcapade/MoCapade, previously the quality reference, was acquired by Epic in February 2026 and its online tools reportedly closed in April 2026.

### Cited Findings

**Open research code — licences and capabilities**
- GVHMR (zju3dv): README documents a static-camera flag (`-s`) and, without it, camera estimation by visual odometry; DPVO is no longer the default and a built-in "SimpleVO" replaced it; a `f_mm` focal-length option was added 2025-03-08, which is the latest news entry on the repo. The README does not mention hands, face, or multi-person. — [GVHMR repo, P](https://github.com/zju3dv/GVHMR)
- GVHMR licence: permission is granted "for educational, research and non-profit purposes only"; modifications must be open-sourced and may not be used commercially; commercial use requires contacting the authors. — [GVHMR LICENSE, P, snippet](https://github.com/zju3dv/GVHMR/blob/main/LICENSE)
- SMPL-X body model licence: non-commercial scientific research, education, or artistic projects only; incorporation in a commercial product, use in a commercial service, or producing artifacts for commercial purposes is excluded. — [SMPL-X model licence, P, snippet](https://smpl-x.is.tue.mpg.de/modellicense.html)
- A community repackage of GVHMR on PyPI exposes three camera modes: static, moving (rotation only), and a VGGT option for moving cameras with world translation. — [gvhmr on PyPI, S, snippet](https://pypi.org/project/gvhmr/)
- PromptHMR: supports video, a multi-human world-coordinate pipeline, moving cameras (with a `--static_camera` flag for locked shots); pipeline components listed include DROID-SLAM and Metric3D; outputs include a GLB that "can be imported into the Blender editor to visualize human and camera motion", plus MCS files and `results.pkl`; training code is withheld "due to licensing agreements". — [PromptHMR repo, P](https://github.com/yufu-wang/PromptHMR)
- A third-party notices file describes PromptHMR's video runtime as covered by a non-commercial scientific research licence (secondhand description, not the upstream licence text). — [Chamiseul_RGBMocap THIRD_PARTY_NOTICES, S, snippet](https://github.com/mayalhc/Chamiseul_RGBMocap/blob/main/THIRD_PARTY_NOTICES.md)
- The author of the main Blender wrapper for these models (GVHMR, PromptHMR, 4D-Humans, HaMeR) states that, as far as he knows, none of the underlying libraries can be used commercially, citing both code and dependencies such as the SMPL model. — [ceb_hubmocap repo, P](https://github.com/carlosedubarreto/ceb_hubmocap)
- SAM 3D Body (Meta, CVPR 2026): single-image full-body mesh recovery using Meta's MHR rig instead of SMPL. Secondary sources say the SAM License permits royalty-free commercial use with restrictions (no military/weapons/ITAR uses, sanctioned parties excluded; Meta may change terms), and that MHR is separately licensed under a permissive commercial licence — read both licence files before commercial use. — [Roboflow licence summary, S](https://roboflow.com/model-licenses/segment-anything-3); [dev.to write-up, S](https://dev.to/arkadiuss/a-3d-body-scan-for-nine-cents-without-smpl-4me1); [HF model card, P](https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/README.md)
- SAM 3D Body weaknesses reported at release: interactions between people and objects, and hand poses. — [3printr, S](https://www.3printr.com/sam-3d-meta-presents-models-for-3d-reconstruction-from-everyday-images-3285492/)
- Human3R (ICLR 2026): feed-forward, online 4D human-scene reconstruction in the world frame from casual monocular video; outputs multiple SMPL-X bodies plus camera; about 15 FPS; authors note coarse human-scene interaction may show penetration. — [Human3R arXiv, P](https://arxiv.org/abs/2510.06219v1)
- JOSH: optimisation-based; jointly optimises camera pose, global human motion and scene from monocular video in one stage (slower than feed-forward methods). — [JOSH arXiv, P](https://arxiv.org/html/2501.02158v2)

**Commercial products**
- Autodesk Flow Studio (formerly Wonder Studio / Wonder Dynamics; renamed after Autodesk's acquisition): markerless AI mocap of body, face and hands exported as FBX; camera tracking with auto-matched camera movement and focal length; exports FBX or USD for Maya, Blender, Unreal, 3ds Max; full Blender/USD scene export including characters, animation, cameras and layout. — [Autodesk Flow Studio product details, P, snippet](https://www.autodesk.com/eu/products/flow-studio/product-details); [Autodesk overview, P, snippet](https://www.autodesk.com/products/flow-studio/overview)
- Flow Studio pricing (third-party tracker, facts checked 2026-11-09): Free $0 / 300 credits; Lite $7/mo / 2,100 credits; Standard $30/mo / 6,000; Pro $64/mo / 12,000 (annual billing); Enterprise custom. AI Mocap 10 credits/s; Live Action 20 credits/s; Camera Track and Clean Plate 7 credits/s each (Lite and above). Unreal export from Lite; USD scene export from Standard; Free is 720p watermarked. The same page's timeline lists Lite $10 / Standard $45 / Pro $95 for mid-2026 (probably monthly billing) — unreconciled; verify on Autodesk's buy page. — [usagepricing.com, S](https://www.usagepricing.com/blueprint/wonder-dynamics); credit rates corroborated by [Autodesk compare page, P, snippet](https://www.autodesk.com/products/flow-studio/compare)
- Flow Studio quality (Aug 2025 comparison by an animation studio): good quality, free tier 30 s/month, but output is camera-relative and it has "issues with moving cameras"; best with a static camera. — [TATO Studio comparison, S](https://tato.studio/blog/best-ai-video-to-mocap)
- Flow Studio earlier limitations: no character-character or character-object interaction support (2023); reviewers describe the mocap as "a little bit janky" with potentially extensive clean-up. Autodesk marketing in 2026 mentions "improved grounding and more natural 3D motion". — [CG Channel 2023, S](https://www.cgchannel.com/2023/09/wonder-studio-now-does-standalone-ai-motion-capture/); [Autodesk overview, P, snippet](https://www.autodesk.com/products/flow-studio/overview)
- Move.ai: Move One is a single-camera iPhone app; official pricing page lists FBX, USD and MP4 export; free users can record up to 30 s. Third-party price figures conflict ($14–18/seat/month for a Starter tier vs "from about $99/month"). One 2026 roundup calls Move.ai the quality leader for markerless capture with strength in multi-camera. The 2025 TATO comparison: accurate on challenging poses, handles hands and knees on the ground better than most, slightly more foot sliding than QuickMagic, more expensive. — [Move One pricing help, P, snippet](https://help.move.ai/en/articles/8851451-move-one-pricing); [aitoolsatlas, S, snippet](https://aitoolsatlas.ai/tools/move-ai/pricing); [3dxdev 2026-08-28, S](https://3dxdev.com/ai-motion-capture-from-video-in-2026-every-tool-compared/); [TATO, S](https://tato.studio/blog/best-ai-video-to-mocap)
- QuickMagic: $9.99/mo for 150 s; smooth and accurate, working finger tracking, some sliding on complex poses such as kneeling (Aug 2025). — [TATO, S](https://tato.studio/blog/best-ai-video-to-mocap)
- DeepMotion: free 25 s; $15/mo for 180 s; twitching and sliding; exports FBX/BVH/GLB with optional physics. — [TATO, S](https://tato.studio/blog/best-ai-video-to-mocap); [3dxdev, S](https://3dxdev.com/ai-motion-capture-from-video-in-2026-every-tool-compared/)
- Plask Motion: free 15 s/day; $18/mo for 10 min (2025) or "Pro about $30/month" (2026); twitching and sliding similar to DeepMotion; supports several people in one video; advertises full body and hand capture with FBX/GLB/BVH export. — [TATO, S](https://tato.studio/blog/best-ai-video-to-mocap); [3dxdev, S](https://3dxdev.com/ai-motion-capture-from-video-in-2026-every-tool-compared/); [Plask pricing, P, snippet](https://plask.ai/pricing)
- Rokoko Vision: free single-camera tier, dual-camera on paid; output "shaky and jumpy", not yet professional level. — [TATO, S](https://tato.studio/blog/best-ai-video-to-mocap); [3dxdev, S](https://3dxdev.com/ai-motion-capture-from-video-in-2026-every-tool-compared/)
- Radical: free tier with one export/year, $20/mo; excellent foot contacts but motion sometimes "loses energy" (too smooth). — [TATO, S](https://tato.studio/blog/best-ai-video-to-mocap)
- Cascadeur video mocap: no finger tracking; motion loses energy; occasional twisting or floating. — [TATO, S](https://tato.studio/blog/best-ai-video-to-mocap)
- BlendCap (Blender add-on, one-time $60): runs locally and offline after an ~11 GB setup on Windows/Linux with an NVIDIA GPU; body, hands, fingers and face from a single ordinary clip; retargets to Rigify, Auto-Rig Pro, CloudRig, Mixamo. The article is published by the same site that discusses the product, and does not name the underlying models or the commercial licence. — [3dxdev 2026-08-28, S](https://3dxdev.com/ai-motion-capture-from-video-in-2026-every-tool-compared/)
- Meshcapade (MoCapade): Epic Games announced the acquisition on 2026-02-18; the team joins Epic's AI Research group for Unreal Engine / MetaHuman; Max Planck Innovation takes over SMPL licensing directly. — [Max Planck Society press release, P](https://www.mpg.de/26082348/max-planck-spin-off-draws-epic-games-to-tubingen); [80.lv, S](https://80.lv/articles/ai-motion-capture-startup-meshcapade-now-part-of-epic-games)
- Meshcapade's online editor and mocap tool (me.meshcapade.com) is reported as shut down on 2026-04-18 by aggregator sites; no Epic/Max Planck statement confirming the date was found. — [aixploria, S](https://www.aixploria.com/en/meshcapade-ai-3d-avatar-human/)
- Last Meshcapade commercial-use price list found for SMPL-family tech: €2,000/year (micro companies) and €10,000/year (SMEs); may be outdated after the acquisition. — [Meshcapade avatar-tech page, P, snippet](https://meshcapade.com/assets/avatar-tech)

**Shared failure modes**
- All single-camera tools share depth ambiguity, temporal jitter, foot sliding, and trouble with fast action, unusual poses and occlusion; "expect a clean-up pass". — [3dxdev 2026-08-28, S](https://3dxdev.com/ai-motion-capture-from-video-in-2026-every-tool-compared/)
- An August 2026 article is devoted specifically to why phone-video mocap fails (occlusion, camera angle, foot contact, retargeting); the page blocked automated access, so only its title is recorded. — [Nerdbot 2026-08-17, S](https://nerdbot.com/2026/08/17/why-phone-video-ai-mocap-fails-occlusion-camera-angle-foot-contact-and-retargeting/)

### Inferences
- For a commercial studio the licence, not the quality, is the first filter. GVHMR, PromptHMR and every SMPL/SMPL-X-based path are excluded unless separate licences are negotiated (GVHMR authors; SMPL via Max Planck Innovation after the Meshcapade acquisition). The SMPL-X licence wording ("producing artifacts for commercial purposes") appears to cover even an intermediate previs that never ships; this needs a legal reading, not an assumption.
- The most plug-and-play commercially licensable candidate for a first experiment is Flow Studio (mocap FBX + camera track + Blender scene in one service), with QuickMagic or Move.ai as body-quality alternatives that do not deliver a camera solve.
- SAM 3D Body is the only strong open model found with a commercially usable licence, but it is per-frame: a studio would have to add temporal smoothing, world trajectory and camera solve itself.
- Film footage (close-ups, waist-up framing, cut-off legs, moving cameras) is exactly where these tools are weakest: commercial tool guidance and the comparison above assume a full-body subject and often a static camera.

### Gaps
- Not verified this session (no page opened): licences and current status of WHAM, TRAM, 4D-Humans/HMR2, TokenHMR, SMPLer-X/SMPLest-X, Multi-HMR, CameraHMR, GENMO; any newer 2026 world-grounded HMR releases beyond Human3R. Each must be checked at its repo before use.
- No official 2026 Move.ai price list could be opened (pricing URLs returned 404); whether Move.ai accepts arbitrary existing footage with a moving camera, and its finger-tracking status, are unconfirmed.
- No information found on Kinetix, Radical's 2026 status, or any Adobe video-mocap offering.
- No source tested any of these tools specifically on film-style close-ups or partial bodies; the weakness is inferred from input guidance.
- Whether BlendCap's underlying models permit commercial use is unknown.

---

## A2. Recovering the camera move from the same footage and bringing it into Blender/Unreal/Maya with the mocap

### Takeaway
Joint human-plus-camera output into a DCC exists in three forms: Flow Studio's camera track/scene export (commercial), PromptHMR's GLB with human and camera (non-commercial), and generic camera estimators (ViPE, VGGT) that need a custom import step. Reliability on shots where a person fills the frame is poorly documented, and the one practitioner comparison found says Flow Studio is best with a static camera.

### Cited Findings
- Flow Studio: camera tracking auto-matches camera movement and focal length and is "export ready"; Camera Track is a separate Wonder Tool at 7 credits/s on Lite and above; a "Video to 3D Scene" feature reconstructs cameras, characters and environments. — [Autodesk product details, P, snippet](https://www.autodesk.com/eu/products/flow-studio/product-details); [usagepricing.com, S](https://www.usagepricing.com/blueprint/wonder-dynamics); [Autodesk overview, P, snippet](https://www.autodesk.com/products/flow-studio/overview)
- Flow Studio practitioner note (Aug 2025): camera-relative output and issues with moving cameras. A separate review flagged "incomplete camera tracking capabilities". — [TATO, S](https://tato.studio/blog/best-ai-video-to-mocap); [futurepedia review, S, snippet](https://www.futurepedia.io/tool/wonder-studio)
- PromptHMR video demo writes a GLB importable into Blender showing both human and camera motion (DROID-SLAM and Metric3D are among its components). — [PromptHMR repo, P](https://github.com/yufu-wang/PromptHMR)
- GVHMR estimates the moving camera with its own SimpleVO (previously DPVO); the README does not describe exporting the camera. — [GVHMR repo, P](https://github.com/zju3dv/GVHMR)
- ViPE (NVIDIA, first released 2025-08; v1.2.0 June 2026 with 2.7x speed-up; ViPE-Long November 2026): estimates intrinsics, camera motion and dense near-metric depth from raw video; NVIDIA says it handles dynamic selfie videos, cinematic shots and dashcams, and is more efficient than MegaSaM because it needs no per-frame optimisation. Output is pose/intrinsics/depth folders (`.npz`), with no COLMAP or Blender exporter mentioned. — [ViPE repo, P](https://github.com/nv-tlabs/vipe); [NVIDIA ViPE page, P, snippet](https://research.nvidia.com/labs/toronto-ai/vipe)
- ViPE licence: NVIDIA-authored code is Apache 2.0, but bundled third-party code keeps its own terms (UniK3D is CC BY-NC-SA 4.0; DPVO MIT; LieTorch BSD-3), downloaded models carry their own licences, and the MetricTrail weights are under an NVIDIA non-commercial licence. — [ViPE repo, P](https://github.com/nv-tlabs/vipe)
- VGGT: code relicensed 2025-07-29 to permit commercial use (excluding military); only the gated `VGGT-1B-Commercial` checkpoint is commercially licensed, the original VGGT-1B weights remain CC BY-NC 4.0. — [VGGT-1B-Commercial on HF, P, snippet](https://huggingface.co/facebook/VGGT-1B-Commercial)
- Open Video Tracker (Blender add-on) automatically imports camera objects with animation, using a photogrammetry pipeline rather than a human-aware method. — [Open Video Tracker docs, S, snippet](https://haseebahmed295.github.io/open_video_tracker_docs/index.html)
- Human3R and JOSH solve camera and humans jointly in a world frame (see A1). — [Human3R, P](https://arxiv.org/abs/2510.06219v1); [JOSH, P](https://arxiv.org/html/2501.02158v2)

### Inferences
- When the actor covers most of the frame, classical feature tracking (Blender tracker, COLMAP/GLOMAP, SynthEyes) has few static features; learned dynamic-scene estimators (ViPE, MegaSaM, VGGT-class) are the tools designed for that case, but no source measured their accuracy on person-dominant handheld/dolly shots.
- For the downstream purpose (a motion/camera reference for Seedance, not a matchmove for compositing), the camera only has to be plausible and smooth, not sub-pixel accurate. A solve that would be unusable for VFX may be adequate here, and the human trajectory and camera only need to be consistent with each other.
- Commercially clean camera options appear to be: Flow Studio's track, VGGT with the commercial checkpoint, or manual/semi-manual tracking; ViPE needs a component-by-component licence check.

### Gaps
- MegaSaM licence and any Blender import path: not found.
- No sources found on SynthEyes, GLOMAP/COLMAP or the Blender tracker specifically for person-dominant shots, nor on TRAM/WHAM camera export.
- No Autodesk help-centre page could be opened describing world-space vs camera-space mocap export, scale conventions, or framing requirements.
- No quantitative report on camera-solve reliability for handheld/dolly shots with a large foreground person.

---

## A3. End-to-end "video → animated mannequin in Blender with matching camera" pipelines and clean-up time

### Takeaway
One-click-style options exist (Flow Studio scene export; CEB HubMocap add-on for GVHMR/PromptHMR/4D-Humans/HaMeR; BlendCap), but only Flow Studio documents a matching camera in the same export, and the free add-on path is explicitly non-commercial. Clean-up estimates are vague — "expect a clean-up pass", with one 2026 guide suggesting an hour or more per clip for production-quality animation.

### Cited Findings
- CEB HubMocap (free Blender add-on by Carlos Barreto; paid packs bundle the Python environments): runs 4D-Humans, GVHMR, PromptHMR and HaMeR as downloadable modules, detects multiple people and lets the user choose one, imports the result into Blender; requires SMPL registration; Windows embedded Python; camera import is not documented. — [ceb_hubmocap repo, P](https://github.com/carlosedubarreto/ceb_hubmocap)
- In the GVHMR issue thread (Sep 2024 – Jan 2026) the add-on author says body comes from GVHMR or PromptHMR, fingers from HaMeR with a smoothing option because hand tracking "jitters", results can be retargeted to a Mixamo character, and some runs used about 15 GB on his 24 GB card. — [GVHMR issue #10, P](https://github.com/zju3dv/GVHMR/issues/10)
- Flow Studio exports a full Blender or USD scene with characters, animation, cameras and layout; USD scene export is on Standard and above. — [Autodesk product details, P, snippet](https://www.autodesk.com/eu/products/flow-studio/product-details); [usagepricing.com, S](https://www.usagepricing.com/blueprint/wonder-dynamics)
- BlendCap: local Blender add-on with retargeting to Rigify/Auto-Rig Pro/CloudRig/Mixamo (see A1). — [3dxdev, S](https://3dxdev.com/ai-motion-capture-from-video-in-2026-every-tool-compared/)
- Clean-up: one 2026 guide for game animation advises budgeting "an hour or more per clip" for clean-up of single-camera AI mocap. — [snippet; from one of these 2026 roundups: [strayspark](https://www.strayspark.studio/blog/ai-mocap-indie-developers-2026-comparison), [quickmatch](https://quickmatch.si/guides/ai-animation-mocap-for-games/)]
- Seedance-side tooling already assumes 3D previs input: Chinese coverage describes Maya/Blender "white-model plug-ins" that export a white-model video with the camera path for upload to Jimeng (plug-in names not given), and the Xiaoyunque handbook describes a built-in "3D director stage" with a character action library, camera keyframes and a timeline that records a reference video for Seedance 2.5. — [smzdm 2026-08-06, S](https://post.smzdm.com/p/anvqol20/); [Xiaoyunque Seedance 2.5 handbook, P, 2026-09-23](https://www.xiaoyunque.com/tutorials/seedance-2-5)

### Inferences
- Cost model for a Flow Studio experiment (derived from the credit rates above): a 10 s shot costs 100 credits for mocap plus 70 for the camera track, i.e. 170 credits; Standard (6,000 credits) covers roughly 35 such shots per month, Lite (2,100) roughly 12.
- The clean-up bar for this use case is lower than for game animation: the mannequin render is only a motion reference, so residual finger noise or minor foot sliding may be tolerated by the generator, but jitter and wrong limb depth will likely be reproduced. The "hour per clip" figure is therefore an upper-bound style estimate, not a measurement for this workflow.
- Because Seedance tooling already ships a 3D director stage with an action library, the missing piece in the market is specifically "import mocap from footage into that kind of stage"; no product documenting that was found.

### Gaps
- No practitioner report quantifying per-shot time for "footage → mocap + camera → Blender mannequin render".
- Cascadeur's 2026 video-mocap capabilities and pricing were not verified beyond the 2025 comparison.
- Whether CEB HubMocap or BlendCap import a solved camera is undocumented.

---

## A4. Retargeting to different proportions and rendering as a neutral mannequin — does it solve silhouette leakage? Any reports of mocap-derived greybox renders fed to Seedance/Kling/VACE/Runway?

### Takeaway
Seedance 2.5 officially supports white-model/greybox reference videos and reviewers call it one of its most stable capabilities, so the downstream half of route A is on firm ground. No source was found that tests mocap-derived mannequins from live footage, or that measures whether a mannequin with the character's proportions removes the actor-silhouette problem; an interesting shortcut exists where Seedance 2.5 itself converts footage into a geometric mocap-figure video.

### Cited Findings
- Dreamina's own documentation: Seedance 2.5 offers reference-driven generation using green-screen footage and "white model (3D-style)" inputs to define motion, spatial layout and character interaction, claimed to make multi-person and directional movement easier to control (vendor claim, article dated 2026-10-24). — [Dreamina how-to, P/vendor](https://dreamina.capcut.com/seedance/how-to-use-seedance-2-5)
- Official handbook guidance: state in the prompt which elements of the white-model video to use, e.g. "参考[视频1]的运镜、动作" for camera and motion only (add lighting only if wanted); map each reference image to the matching figure in the white model; another example takes only fight motion and shot rhythm from a video and excludes its character appearance. Input limits: up to 30 images, up to 10 video clips totalling 30 s or less, up to 10 audio clips totalling 30 s or less. — [Xiaoyunque Seedance 2.5 handbook, P](https://www.xiaoyunque.com/tutorials/seedance-2-5)
- Pixo's blockout guide (published 2026-08-20, updated 2026-11-02): coarse blockout (spheres, cylinders, boxes) is "better today" than fine blockout; motion/blocking, camera (dolly, crane, orbit), composition/occlusion and timing transfer; in coarse mode limbed stand-ins go stiff unless the full limb action sequence is written in the prompt; viewport helpers (trajectory lines, axes, camera cones) must be removed or they appear in the render; a single grey mannequin trudging clip was rendered into four differently styled segments with consistent performance; limits quoted as up to 30 s at 720p. — [Pixo blog, S](https://pixo.video/blog/seedance-previz-blockout)
- Chinese hands-on coverage: a white humanoid walking reference was used to drive a small dog that kept the walk and the original camera move; a review rates white-model reference among the more stable capabilities and a likely standard usage; another review notes prompt moderation is unstable and multimodal understanding still has room to improve. — [Sohu, S, snippet](https://timeline.sohu.com/news/ODsqvNBI4v?from=news); [elsewhere.news, S, snippet](https://elsewhere.news/en/crossing/seedance-25-20-25); [TMTPost, S, snippet](https://www.tmtpost.com/8088164.html)
- Generative "mocap visualisation": Kapwing's AI Motion Capture tool uploads a live video, selects Seedance 2.5 and uses a prompt to replace the person with a geometric figure (spheres for joints, cylinders/capsules for limbs, matte materials, plain background, no clothing or props), then offers that clip as a reusable motion reference for motion transfer onto a character image; it claims pacing, hand gestures and facial cues carry over when visible. — [Kapwing AI Motion Capture, vendor](https://www.kapwing.com/ai/motion-capture)
- Blender → Wan pipelines exist for the open-model side: a RunComfy workflow drives Wan 2.1 VACE 14B from Blender depth/outline/pose passes, and Playbook 3D publishes a "Wan 2.2 Graybox to Rendered Sequence" workflow (Flux Kontext depth LoRA + Wan 2.2 Fun Control). — [RunComfy workflow, S](https://www.runcomfy.com/comfyui-workflows/blender-to-comfyui-ai-renderer-2-0-workflow-cinematic-video-output); [Playbook 3D, S](https://playbook3d.gumroad.com/l/vpjph)
- On shape cues in 3D-body motion signals: the Wan-Animate paper explains it chose skeletons over SMPL renders because SMPL renders "can cause the model to rely on the shape cues embedded within the motion signal". — [Wan-Animate paper, P](https://arxiv.org/html/2509.14055)

### Inferences
- A generator that follows a mannequin video follows its silhouette too (the Wan-Animate authors say so about SMPL renders, and Pixo's fine-mode prompt needed "no deformation or size change"). So a mannequin does not remove shape conditioning; it replaces the actor's shape with whatever shape the mannequin has. Retargeting to a mannequin built with the 2D character's proportions is therefore the step that should fix leakage, and it is the step with no published evidence — this is the core thing the experiment has to test.
- The mannequin route should fix the shimmer problem by construction, because a rendered mannequin is temporally stable, unlike per-frame depth estimation.
- Pixo's warning that limbed stand-ins go stiff in coarse mode suggests a mocap mannequin should be treated as a "fine" blockout (clear, complete figure) with the prompt naming the performance beats, rather than as a coarse stand-in.
- Retargeting to big-head/short-limb proportions will break contacts captured on the real actor (hand-to-face, hand-to-prop, foot placement, two-character contact), so IK/contact correction is likely the main manual cost; no source quantifies it.
- The Kapwing-style approach (ask Seedance 2.5 itself to turn the footage into a geometric figure, optionally with stylised proportions, then use that as the reference) is a cheap "route A-lite" worth testing next to true mocap: no licences, no 3D, but a second generation with its own drift, and proportions controlled only by prompt.

### Gaps
- No report found of anyone feeding mocap-derived (from live footage) mannequin renders into Seedance, Kling, Wan VACE or Runway; all white-model examples found use hand-animated or library-animated previs.
- No test found of 2D/anime output specifically from white-model references in public sources (examples are photoreal, sci-fi or fantasy).
- No evidence on how mannequin colour, detail level, facial features or head-size exaggeration affect Seedance's result.

---

## A5. What is lost in route A, and how to compensate

### Takeaway
Body trajectory and camera survive; facial performance, reliable fingers, cloth/hair, prop and environment interaction and multi-character contact mostly do not, and published guidance for compensating is limited to prompt-level instructions.

### Cited Findings
- Finger capture from video is available but noisy: HaMeR-based fingers "jitter" and need smoothing; QuickMagic has working finger tracking; Cascadeur has none. — [GVHMR issue #10, P](https://github.com/zju3dv/GVHMR/issues/10); [TATO, S](https://tato.studio/blog/best-ai-video-to-mocap)
- Flow Studio exports face tracking as part of a Blender scene and captures body, face and hands. — [Autodesk product details, P, snippet](https://www.autodesk.com/eu/products/flow-studio/product-details)
- Object and multi-person interaction are documented weak points: SAM 3D Body (people-object interaction, hands), Flow Studio (no character-character or character-object interaction in early versions; struggles with physical interactions and heavy occlusion), Human3R (penetration in human-scene interaction). — [3printr, S](https://www.3printr.com/sam-3d-meta-presents-models-for-3d-reconstruction-from-everyday-images-3285492/); [CG Channel, S](https://www.cgchannel.com/2023/09/wonder-studio-now-does-standalone-ai-motion-capture/); [Human3R, P](https://arxiv.org/abs/2510.06219v1)
- Mocap from video tends to "lose energy" (over-smoothing) in several tools. — [TATO, S](https://tato.studio/blog/best-ai-video-to-mocap)
- Seedance-side compensation documented so far is prompt-based: with limbed stand-ins, write the full limb action sequence; add explicit constraints such as "no clipping, no face drift"; describe in text any subject that has no image/video reference. — [Pixo blog, S](https://pixo.video/blog/seedance-previz-blockout); [Xiaoyunque handbook, P](https://www.xiaoyunque.com/tutorials/seedance-2-5)

### Inferences
- A mannequin without a face gives the generator no expression timing, so facial acting has to come from the prompt (the studio's existing acting-system prompts) or from a second reference. Since the handbook allows up to 10 reference clips and lets the prompt assign roles per clip, a possible compensation is mannequin video for body/camera plus a separate, non-identifying face-timing reference — untested in any source found.
- The current depth-map method keeps props, cloth and environment geometry for free; route A drops them unless they are blocked in by hand, so route A is strongest for single-character performance shots and weakest for prop-heavy or contact-heavy shots.

### Gaps
- No source evaluates a "face landmarks pass" or similar auxiliary reference with Seedance 2.5.
- No source measures how well Seedance invents secondary motion (hair, cloth) from a bare mannequin versus from depth.

---

## B1. Capabilities, limits, pricing and licences of dedicated motion-transfer models

### Takeaway
Three generations now coexist: skeleton-driven (Wan 2.2 Animate, SCAIL, VACE pose), end-to-end "raw driving video" models (SCAIL-2 in June 2026, Wan-Animate-2 in August 2026, DreamActor M2.0), and closed services (Kling Motion Control 2.6/3.0, Runway Act-Two, Luma Ray3 Modify). The Apache-2.0 open models fit a 24 GB card in quantised form; only Kling documents following the driving video's camera, and most others advise a locked camera.

### Cited Findings

**Wan 2.2 Animate (Alibaba, weights + code released 2025-09-19, Apache 2.0)**
- Two modes: animation (character image mimics the motion in the input video) and replacement (character replaces the person in the video); `--retarget_flag` for pose retargeting in animation mode; `--use_relighting_lora` in replacement mode; LoRAs trained on Wan 2.2 are not recommended with it. — [HF model card, P](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- Method: body via ViTPose skeletons added to noise latents; face via raw face crops encoded to a 1D latent and injected by cross-attention every 5 layers; animation mode preserves the background of the source image; the paper states only that "the model can generate character actions that include camera movements" without a mechanism; long videos are generated in 78-frame segments with 1 or 5 guidance frames. — [Wan-Animate paper, P](https://arxiv.org/html/2509.14055)
- Stated limitations: 2D skeletons are sensitive to posture when estimating bone lengths and suffer missing/erroneous keypoints; replacement with large body-shape differences may deform. — [Wan-Animate paper, P](https://arxiv.org/html/2509.14055)
- Community notes: internal looping degrades noticeably after about three extensions; identity is weaker than VACE 2.1 though lighting is impressive; adding "human" to the negative prompt stops the model giving a non-human character a human face. — [wanx-troopers notes, S](https://wanx-troopers.github.io/wan-animates.html)

**Wan-Animate-2 (Alibaba Tongyi Lab, released 2026-08-07, Apache 2.0)**
- Consumes the driving video directly in a redesigned dual-branch DiT with no intermediate skeleton or expression extractor; claims to animate stylised cartoon characters, animals and robots; offers text-driven viewpoint control that "decouples the output camera perspective from the driving video" via a Viewpoint LoRA trained on synthetic Unreal data with 48 discretised viewpoints; supports multi-character driving. — [alphaXiv abstract page, P](https://www.alphaxiv.org/abs/2608.06009); [ComfyUI-Wiki news, S](https://comfyui-wiki.com/en/news/2026-08-07-wan-animate-2)
- Base and Distillation weights are public; the real-time Lite variant (24 fps at 400×720 on a 4-GPU setup) was announced but not released. Native ComfyUI support via experimental nodes `WanAnimate2ToVideo` (with pose strength / start / end and reference strength controls) and `WanAnimate2Cache` (roughly halves generation time); distilled model runs 10 steps without CFG, base 40 steps. — [ComfyUI-Wiki news, S](https://comfyui-wiki.com/en/news/2026-08-07-wan-animate-2); [HF model card, P, snippet](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B)
- Local sizing: full bf16 file 32.79 GB; on 24 GB the official `wan_animate_2_int8_convrot.safetensors` (16.65 GB) is the suggested choice; GGUF quants go down to 12 GB cards but the stock GGUF loader silently loads the model as plain Wan 2.1 I2V and ignores the driving video unless a loader with `model_type=animate2` is used; the biggest quality lever is a reference image whose pose matches the driving video's opening pose. The guide did not measure speed on consumer cards, and did not evaluate identity leakage, camera motion or anime characters. — [localaimaster guide, S](https://localaimaster.com/blog/wan-animate-2-local-guide)
- One reported setup peaked at 21.4 GB at 480×832, 81 frames on an RTX 4090; the official README's default 720p configuration targets 8×A800. — [snippet; likely [nemovideo review](https://www.nemovideo.com/blog/ai-video-models/wan-animate-2-comfyui-review)]

**SCAIL and SCAIL-2 (Zhipu / zai-org)**
- SCAIL (arXiv Dec 2025): 14B DiT using in-context learning of a 3D-consistent pose representation; Apache 2.0; native in ComfyUI from 2026-03-01; pose extraction partly ported in ComfyUI-SCAIL-Pose without multi-character tracking. — [SCAIL repo, P, snippet](https://github.com/zai-org/SCAIL); [arXiv, P](https://arxiv.org/abs/2512.05905)
- SCAIL practitioner notes: retargeting requires NLF + DWPose + reference DWPose inputs; the NLF pose model carries a research-only licence; background must stay nearly static because camera motion causes drift; output goes dark unless the pose video is exactly half the generation resolution. — [wanx-troopers notes, S](https://wanx-troopers.github.io/wan-animates.html)
- SCAIL-2 (June 2026): end-to-end on Wan 2.1 14B, no intermediate pose; replacement and animation modes; SAM 3 tracking; 81-frame segments with 5 overlap frames, each segment queued manually for long videos; native ComfyUI nodes. — [ComfyUI docs, P](https://docs.comfy.org/tutorials/video/zai/scail2.md); [ComfyUI-Wiki news, S, snippet](https://comfyui-wiki.com/en/news/2026-06-09-scail-2-character-animation)
- SCAIL-2 practitioner notes: distorts at untrained dimensions (512×864 or 704×1280 recommended); the reference image must match the driving video's size and position in the first frame; replacement mode is about 3–5× slower than a normal generation; multi-reference was not in training. Third-party VRAM figures are inconsistent (GGUF quants about 6–17 GB); licence listed as Apache 2.0 by a third party. — [wanx-troopers notes, S](https://wanx-troopers.github.io/wan-animates.html); [Floyo, S, snippet](https://floyo.ai/models/scail-2)

**Wan VACE 2.1**
- Supports reference-to-video, video-to-video, masked editing, pose transfer and depth control; one secondary source lists the 14B model as Apache 2.0 (model card not opened). VRAM guidance conflicts: about 20 GB at 720p and about 40 minutes on a 4090 in one guide, "48 GB or more for stable operation" in another, quantised GGUF fitting easily in 24 GB in a third. — [ComfyUI VACE docs, P, snippet](https://docs.comfy.org/tutorials/video/wan/vace); [HF model card, P, not opened](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B); [snippet, sources among: [Banodoco KB](https://nathanshipley.github.io/banodoco-kb/kb/wan/index.html), [RunPod](https://www.runpod.io/blog/the-dos-and-donts-of-vace)]

**Kling Motion Control (Kuaishou; closed, cloud)**
- Official guide (dated 2026-03-05) covers 2.6 and 3.0. Motion reference: 3–30 s, one continuous shot, whole body and head visible, one character, real humans recommended, moderate speed; output length matches the upload. Default mode "Character Orientation Matches Video": movement, expressions, camera movement and orientation follow the reference. Mode "Character Orientation Matches Image": movement and expressions follow the reference while camera moves are set by prompt. The same guide also advises a reference with no cuts or camera movement. — [Kling Motion Control user guide, P](https://kling.ai/quickstart/motion-control-user-guide)
- Pricing in the official guide: 3.0 Professional 12 credits/s, Standard 9 credits/s; 2.6 Professional 8 credits/s, Standard 5 credits/s; billed per rounded second. Resellers quote roughly $0.10–0.16 per second for 3.0. — [Kling guide, P](https://kling.ai/quickstart/motion-control-user-guide); [EvoLink, S, snippet](https://evolink.ai/kling-v3-motion-control)
- API-level limits for 3.0: image-orientation mode caps output at 10 s, video-orientation at 30 s. — [Vercel AI Gateway listing, S, snippet](https://vercel.com/ai-gateway/models/kling-v3.0-motion-control)
- 3.0 improves facial consistency across angles and long motions and adds element binding (uses facial information only; works only in video-orientation mode). — [Kling guide, P](https://kling.ai/quickstart/motion-control-user-guide)
- Kling-MotionControl technical report (arXiv 2026-03-03): unified DiT with separate motion representations for body, face and hands; "adaptive identity-agnostic learning" for retargeting "from realistic humans to stylized cartoons"; over 10× speed-up by distillation. — [arXiv 2603.03160, P](https://arxiv.org/abs/2603.03160)

**Runway Act-Two, Luma Ray3 Modify, DreamActor, OmniHuman**
- Runway Act-Two: transfers head, face, torso and hand performance to a character image or video; with a character video the original camera motion and environment are kept, with a character image only subtle environmental motion such as handheld shake is added; gesture control only with image inputs; outputs up to 10 s at 24 fps; hands and bodies can look robotic or distorted. A vendor-affiliated blog says it explicitly supports cartoons, animals and different proportions. API rate reported as 5 credits/s with a 3 s minimum. — [Atlabs comparison, S](https://www.atlabs.ai/blog/runway-act-two-vs-luma-modify-video); [DualView, S](https://www.dualview.ai/blog/ai-tools/ai-video-editing-models.html); [aiwiki, S, snippet](https://www.aiwiki.ai/wiki/runway_act_two)
- Luma Ray3 Modify (Dec 2025): places a character reference over an actor's performance while keeping original motion and facial animation; up to 30 s output with a 10 s upload limit per one source (another says outputs up to 10 s); 720p/1080p; paid Dream Machine plans; a 2026 source mentions Ray3.2 with video modification and performance transfer. — [DualView, S](https://www.dualview.ai/blog/ai-tools/ai-video-editing-models.html); [Atlabs, S](https://www.atlabs.ai/blog/runway-act-two-vs-luma-modify-video)
- DreamActor M2.0 (ByteDance; arXiv 2601.21716; SIGGRAPH 2026): pose-free, in-context learning on raw RGB frames; one reference image plus one driving video; copies body motion, expression and lip sync while preserving identity and the image background; claimed for humans, cartoons, animals, mascots; weights closed; BytePlus API from about February 2026 at $0.05 per output second (as of Aug 2026); guidance says to match framing between reference and performance and to use a locked camera in the driving video. — [invideo article 2026-08-06, S/vendor](https://invideo.io/blog/dreamactor-motion-control/); listed on [BytePlus model hub, P, snippet](https://ai.byteplus.com/en/model)
- OmniHuman 1.5 is listed on BytePlus as an audio-driven avatar model with reseller prices of about $0.14–0.32/s; it is not a driving-video body-motion transfer tool. — [BytePlus model hub, P, snippet](https://ai.byteplus.com/en/model); [AIMLAPI, S, snippet](https://aimlapi.com/models/omnihuman-v1-5)
- Dreamina's "Motion Control" page names Seedance 2.5 as its engine rather than a separate DreamActor/OmniHuman model. — [Dreamina Motion Control page, vendor, snippet](https://dreamina.capcut.com/ai-video/motion-control)

**Other open animators**
- One-to-All: about 15 minutes on a 4090 at 576×1024, about 2 minutes with lightx2v (2025-12-08); a 613-frame generation worked but burned with lightx2v. Steady Dancer (Nov 2025): a user judged UniAnimate similar or better. — [wanx-troopers notes, S](https://wanx-troopers.github.io/wan-animates.html)

### Inferences
- Camera from the driving video: Kling (video-orientation mode) is the only product whose official documentation says camera movement follows the reference, and even it recommends a camera-static reference. Wan 2.2 Animate keeps the reference image's background; Wan-Animate-2 deliberately decouples viewpoint from the driving video; SCAIL and DreamActor guidance asks for a locked camera. For shots where camera work must be preserved, dedicated animators are weaker than Seedance's reference-video route or route A.
- On a single RTX 4090, Wan-Animate-2 (int8 file, distilled 10-step) and SCAIL-2 (quantised) are the realistic local candidates, at roughly 480p–720p and 81-frame segments, with multi-segment extension for longer shots.
- Because end-to-end models read raw pixels of the driving video, they trade skeleton errors for a new leakage channel (see B2).

### Gaps
- No verified 2026 information found on Viggle, Animate Anyone 2, MimicMotion, UniAnimate-DiT, HunyuanVideo-based animators, or Runway Aleph beyond name mentions; their licences and local requirements were not checked.
- No source documents Wan-Animate-2 or SCAIL-2 generation times on a 4090, nor Wan-Animate-2 behaviour with anime characters or camera motion.
- Whether Dreamina/Jimeng exposes DreamActor M2.0 in its consumer app is unconfirmed.
- Wan 2.2 Animate VRAM figures exist only as an image chart on the model card (unreadable in this session).
- Runway's and Luma's own documentation pages were not opened; all figures for them are secondary.

---

## B2. How these models deal with identity/shape leakage from the driving video

### Takeaway
Skeleton-based models handle shape by retargeting bone lengths to the reference character and handle face identity by compressing face crops; end-to-end models avoid skeleton errors but practitioners report they copy facial shape from the driving pixels, with manual masking as the workaround. Every vendor's practical advice is the same: match framing and proportions between the driving video and the character image.

### Cited Findings
- Wan-Animate retargeting: computes per-limb length ratios between reference and source skeletons, scales bones and translates the pose using a framing anchor (feet for full body, neck for half body); optional alternative normalises both characters to a T-pose with an image-editing model before computing scales; not recommended in replacement mode because it breaks scene interaction. — [Wan-Animate paper, P](https://arxiv.org/html/2509.14055)
- Wan-Animate face leakage control: face crops are compressed to a 1D latent with scaling, colour-jitter and noise augmentations to prevent "identity leakage". — [Wan-Animate paper, P](https://arxiv.org/html/2509.14055)
- Community: proportion mismatch between pose source and character is "a common issue with this series of models"; Kijai's One-to-All pose-detection node reportedly avoids severe limb stretching; Kijai's retargeting node recentres the stick figure on the character's position in the reference frame. — [wanx-troopers notes, S](https://wanx-troopers.github.io/wan-animates.html)
- SCAIL-2 (end-to-end): without intervention it "adopts the driving mouth and nose shape too closely"; painting over faces in the driving video (mask drawn and passed to SAM 3) improves identity persistence. — [wanx-troopers notes (user ucren), S](https://wanx-troopers.github.io/wan-animates.html)
- Kling 2.6 official guidance: the character's proportions and framing (full-body or half-body) should match the motion reference; stylised humanoid characters and "certain non-realistic body proportions can be recognized", though real human motion is recommended; a face very different from a human face may reduce facial quality. — [Kling guide, P](https://kling.ai/quickstart/motion-control-user-guide)
- Third-party Kling guides: significant proportion mismatch produces distortion most visible in arm and leg length; body warping usually comes from framing or aspect-ratio mismatch. — [Atlas Cloud Kling guide, S, snippet](https://www.atlascloud.ai/blog/tips/kling-ai-motion-control); [Oakgen guide, S, snippet](https://oakgen.ai/blog/kling-motion-control-guide)
- Kling's technical report claims natural retargeting across large morphological differences via identity-agnostic learning; one of its comparisons reportedly shows a competing product altering a child character's limb proportions. — [arXiv 2603.03160, P](https://arxiv.org/abs/2603.03160)
- DreamActor M2.0 trains on self-bootstrapped pseudo cross-identity pairs to separate motion from identity. — [invideo article, S](https://invideo.io/blog/dreamactor-motion-control/)

### Inferences
- For a big-head/short-limb 2D character, skeleton retargeting by bone-length ratio handles limb length but not head-size or volume differences, and 2D bone-length estimation is unreliable in foreshortened poses (the paper's own limitation). Expect partial, not complete, removal of actor proportions.
- The SCAIL-2 face-masking trick is conceptually the same move as the studio's depth-map trick: remove identity-bearing pixels from the driving signal before the generator sees them.
- No vendor claims robust results for extreme (chibi-like) proportions; this must be tested per character.

### Gaps
- No independent test found of Kling 3.0, Wan-Animate-2 or DreamActor M2.0 on anime characters with non-realistic proportions.
- How Wan-Animate-2 separates identity from the raw driving video is not described in the material that could be read.

---

## B3. How studios combine dedicated animators with a general model (two-stage approaches), and evidence it helps

### Takeaway
Two-stage chaining is common practice inside the open-model community (SCAIL → Wan Animate; Wan Animate → depth → VACE) and is sold as a product pattern (Seedance-generated mocap figure → motion transfer), but no public evidence was found for the specific chain "dedicated animator rough → Seedance/Kling restyle", nor any controlled comparison showing a quality gain.

### Cited Findings
- Hybrid chain reported by Juan Gea (2025-12-17): human video → SCAIL → Wan Animate when character proportions differ strongly from the source; SCAIL does the retargeting, Wan Animate keeps the background more consistent; adding Uni3C with the original video can keep the camera stable and stop the character sliding. — [wanx-troopers notes, S](https://wanx-troopers.github.io/wan-animates.html)
- Hybrid chain (2026): generate with Wan Animate's internal looping, extract a depth map from that output, then generate the final video with VACE and context windows; a user reported no visible burning. — [wanx-troopers notes, S](https://wanx-troopers.github.io/wan-animates.html)
- Productised two-stage pattern: Kapwing generates a simplified mocap-figure video with Seedance 2.5 and then feeds it to its motion-transfer tool with a character image. — [Kapwing AI Motion Capture, vendor](https://www.kapwing.com/ai/motion-capture)
- Seedance's own handbook supports role assignment per reference (e.g. take only motion and shot rhythm from a video, appearance from images), which is the mechanism a second-stage restyle would rely on. — [Xiaoyunque handbook, P](https://www.xiaoyunque.com/tutorials/seedance-2-5)

### Inferences
- A plausible two-stage design for this studio: stage 1 uses a local Apache-2.0 animator (Wan-Animate-2 or SCAIL-2 with face masking) to produce a rough clip in which the 2D character, with its own proportions, performs the actor's motion; stage 2 feeds that rough to Seedance 2.5 as the reference video with the anchor shot. The rough would already carry the right silhouette, so leakage of the actor's shape would be decided in stage 1. This is a hypothesis; no source reports doing it.
- Stage-1 animators mostly do not reproduce the driving camera (B1), so this chain risks losing the camera work that the depth-map method currently preserves, unless camera is supplied separately (e.g. Uni3C in the Wan stack, or a second reference clip for camera in Seedance).
- Each generative stage adds drift and cost; the Kapwing pattern and the Wan-community chains show the idea is workable, not that it is better.

### Gaps
- No studio case study, breakdown or forum thread found describing Wan Animate (or Kling Motion Control) output being restyled through Seedance or Kling.
- No A/B evidence for one-stage versus two-stage quality.

---

## B4. Head-to-head user comparisons (2026): Seedance 2.x reference-video motion transfer vs Kling Motion Control vs Wan Animate for anime/2D characters

### Takeaway
No rigorous head-to-head test of these three on anime/2D motion transfer was found; available comparisons are third-party blogs that contradict each other and cover mixed model generations.

### Cited Findings
- Blog comparisons split on anime: some name Kling 3.0 (Omni) best for anime storytelling or hand-drawn looks, others favour Seedance for its reference capacity. — [Screenweaver, S, snippet](https://www.screenweaver.ai/blog/best-ai-video-model-anime); [Elser, S, snippet](https://www.elser.ai/blog/kling-vs-seedance-vs-veo-for-anime-videos-which-ai-model-wins-in-2026)
- On motion transfer specifically, one source says Kling Motion Control replicates dance references precisely while Seedance interprets them loosely; another says Seedance beats Kling 3.0 on face fidelity and precise motion control — a direct contradiction. — [snippets from comparisons including [Alici](https://alici.ai/blog/best-seedance-2-alternatives-2026) and [Kenerate](https://kenerateai.com/blogs/seedance-vs-kling-vs-veo-vs-wan-vs-runway-2026)]
- Wan coverage in these comparisons is thin and mostly about the older Wan 2.2 Animate: open and local, but judged below Kling and Seedance on output quality, with the caveat that Wan plus the right LoRA can win on stylised work. Newer comparisons note Wan 3.0 and Seedance 2.5 have superseded Wan 2.7 and Seedance 2.0. — [aifreeapi comparison, S, snippet](https://www.aifreeapi.com/en/posts/wan-2-7-vs-kling-3-0-vs-seedance-2-0)
- Seedance 2.0 blocked direct uploads of real faces, which does not apply to illustrated characters. — [snippet from the same group of comparisons, e.g. [Screenweaver](https://www.screenweaver.ai/blog/best-ai-video-model-anime)]
- Seedance 2.5 is described as a July 2026 ByteDance release generating up to 30 s natively with up to 50 multimodal references. — [Atlas Cloud model page, S, snippet](https://www.atlascloud.ai/models/seedance-2.5)
- A dedicated "Kling Motion Control vs Seedance 2.0: real limits" article exists but returned 404 when opened. — [vidcella, S, unavailable](https://vidcella.ai/posts/kling-motion-control-vs-seedance-2-best-motion-transfer)

### Inferences
- The contradiction between blogs is consistent with the two tools doing different jobs: Kling Motion Control is a dedicated performance-copy model with strict input rules (single continuous shot, whole body, human driver), while Seedance's reference-video mode is a general multimodal conditioner that also carries camera, timing and staging. For film footage with camera moves and partial bodies, Seedance's mode matches the input better; for a clean full-body performance, Kling's is the more literal copier.
- An in-house comparison on the studio's own shots is the only way to settle this; a minimal test matrix is: (1) current depth-map reference, (2) retargeted mannequin reference, (3) Seedance-generated geometric figure reference, (4) Kling 3.0 Motion Control in video-orientation mode, (5) Wan-Animate-2 or SCAIL-2 locally — judged on silhouette fidelity to the character, camera fidelity, flicker and facial acting.

### Gaps
- No Reddit, Bilibili, Zhihu or Korean-language head-to-head on anime motion transfer could be retrieved in this session (search returned no such threads); community sentiment is therefore under-sampled.
- No comparison includes Seedance 2.5 reference-video transfer against Kling 3.0 Motion Control and Wan-Animate-2 on the same clip.
