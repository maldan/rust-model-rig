rig.load_model("F:/3d/h_project/New/Miruko/Miruko.glb")

-- rig.load_model("F:/csharp/VR_Waifu/asset/model/Miruko/Miruko.gltf")
print("loaded " .. #rig.list_bones() .. " bones")

local function bone(name, side)
  return name .. "." .. side
end

local function clavicle_follow(side)
  local d = rig.create_driver("clavicle " .. side)
  local get = d:add("bone_get", { pos = {980, 570}, bone = bone("Arm", side), space = "offset" })
  local scale = d:add("quat_scale", { pos = {1316, 610}, t = 0.8 })
  local set = d:add("bone_set", { pos = {1540, 645}, bone = bone("Shoulder", side), space = "offset" })
  d:link(get, "rot", scale, "q")
  d:link(scale, "out", set, "rot")
end

local function scapula_slide(side)
  local d = rig.create_driver("scapula " .. side)
  local get = d:add("bone_get", { pos = {935, 635}, bone = bone("Arm", side), space = "offset" })
  local euler = d:add("quat_to_euler", { pos = {1290, 660} })
  local map = d:add("map_range", { pos = {1600, 695}, in_from = 0, in_to = 90, out_from = 0, out_to = 0.04 })
  local xyz = d:add("combine_vec3", { pos = {1865, 802} })
  local set = d:add("bone_set", { pos = {2130, 845}, bone = bone("Scapula", side), space = "offset" })
  d:link(get, "rot", euler, "q")
  d:link(euler, "z", map, "in")
  d:link(map, "out", xyz, "x")
  d:link(map, "out", xyz, "y")
  d:link(map, "out", xyz, "z")
  d:link(xyz, "out", set, "pos")
end

for _, side in ipairs({ "L", "R" }) do
  rig.create_ik(bone("Hand", side), 2)
  rig.create_soft(bone("Boob", side), {
    gravity = 9.8,
    stiffness = 55,
    damping = 5,
    inertia = 0.1,
    max_angle = 60,
  })
  clavicle_follow(side)
  scapula_slide(side)
end

local d = rig.create_driver("Driver 5")
local n1 = d:add("bone_get", { pos = {1020, 540}, bone = "Thigh.R", space = "offset" })
local n2 = d:add("bone_set", { pos = {2155, 690}, bone = "Ass.R", space = "offset" })
local n6 = d:add("quat_to_euler", { pos = {1370, 565} })
local n7 = d:add("map_range", { pos = {1615.2682, 568.3824}, in_from = -90, in_to = 90, out_from = 0.03, out_to = -0.03 })
local n8 = d:add("combine_vec3", { pos = {1898.2281, 586.7424} })
d:link(n1, "rot", n6, "q")
d:link(n7, "out", n8, "y")
d:link(n8, "out", n2, "pos")
d:link(n6, "z", n7, "in")
d:link(n7, "out", n8, "z")
