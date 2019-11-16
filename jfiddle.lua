#!/usr/local/bin/luajit -i

if IS_FIDDLE then return end
local ok = pcall(dofile, '../include.lua')
if not ok then pcall(dofile, 'include.lua') end

-- Important libraries in the global space
mp = require'msgpack.MessagePack'
si = require'simple_ipc'
local libs = {
	'ffi',
	-- 'torch',
	'util',
	'vector',
	'Body',
}
-- Load the libraries
for _,lib in ipairs(libs) do
	local ok, lib_tbl = pcall(require, lib)
	if ok then
		_G[lib] = lib_tbl
	else
		print("Failed to load", lib)
		print(lib_tbl)
	end
end

-- FSM communicationg
fsm_chs = {}
for sm, en in pairs(Config.fsm.enabled) do
	local fsm_name = sm..'FSM'
	local ch = en and si.new_publisher(fsm_name.."!") or si.new_dummy()
	_G[sm:lower()..'_ch'] = ch
	fsm_chs[fsm_name] = ch
end

-- Shared memory
local listing = unix.readdir(HOME..'/Memory')
local shm_vars = {}
for _,mem in ipairs(listing) do
	local found, found_end = mem:find'cm'
	if found then
		local name = mem:sub(1,found_end)
		table.insert(shm_vars,name)
		require(name)
	end
end

-- Local RPC for testing
rpc_ch = si.new_requester'rpc'
dcm_ch = si.new_publisher'dcm!'
state_ch = si.new_publisher'state!'

--print(util.color('FSM Channel', 'yellow'), table.concat(fsm_chs, ' '))
--print(util.color('SHM access', 'blue'), table.concat(shm_vars,  ' '))

local function gen_screen(name, script, ...)
	local args = {...}
	return table.concat({
			'screen',
			'-S',
			name..table.concat(args),
			'-L',
			'-dm',
			'luajit',
			script
		},' ')
end
function pkill(name, idx)
	if tostring(idx) then
		local ret = io.popen("pkill -f "..name)
	else
		local ret = io.popen("pkill -f "..name..' '..tostring(idx))
	end
end

-- Start script
local runnable = {}
for _,fname in ipairs(unix.readdir(HOME..'/Run')) do
	local found, found_end = fname:find'_wizard'
	if found then
		local name = fname:sub(1,found-1)
		runnable[name] = 'wizard'
	end
end
for _,fname in ipairs(unix.readdir(ROBOT_HOME)) do
	local found, found_end = fname:find'run_'
	local foundlua, foundlua_end = fname:find'.lua'
	if found and foundlua then
		local name = fname:sub(found_end+1, foundlua-1)
		runnable[name] = 'robot'
	end
end
function pstart(scriptname, idx)
	local kind = runnable[scriptname]
	if not kind then return false end
	local script = kind=='wizard' and scriptname..'_wizard.lua' or 'run_'..scriptname..'.lua'
	if tostring(idx) then
		script = script..' '..tostring(idx)
	end
	pkill(script)
	unix.chdir(kind=='wizard' and HOME..'/Run' or ROBOT_HOME)
	local screen = gen_screen(scriptname, script, idx)
	print('screen', screen)
	local status = os.execute(screen)
	unix.usleep(1e6/4)
	local ret = io.popen("pgrep -fla "..script)
	for pid in ret:lines() do
		if pid:find('luajit') then
			return true
		end
	end
end

IS_FIDDLE = true

if arg and arg[-1]=='-i' and jit then
	if arg[1] then
		-- Test file first
		dofile(arg[1])
	end

    -- hardcoded script @befreor
    local block_xpos = wcm.get_cubes_xpos()
    local block_ypos = wcm.get_cubes_ypos()

		-- BLOCK ID (SIM) to BLOCK ID (REAL)
		-- for accessing 'final_block_rel_loc'
		-- e.g. to access BLOCK ID '3' (SIM)
		-- 			final_block_rel_loc[block_id_map[3]]
		local block_id_map = {1, 5, 7, 4, 2, 3, 6}

		-- [TODO] INITIAL 'X' LOCATION
		local x_hold_offset = 0.04

		local refine_cubes_xpos = {}
		for i=1,7 do
			table.insert(refine_cubes_xpos, block_xpos[i] - x_hold_offset)
		end

		print("refine_cubes_xpos")
		for i=1,7 do
			print(refine_cubes_xpos[i])
		end

		-- [TODO] RELATIVE/FINAL LOCATION
		-- relative loc based on BLOCK ID (REAL) '0'
		local final_block_rel_loc = { {0,0,0},
														{-3 * 0.025, 1 * 0.025, 0},
														{-3 * 0.025, -3 * 0.025, 0},
														{-1 * 0.025, -2 * 0.025, 0},
														{-2 * 0.025, 0, 0},
														{0, -3 * 0.025, 0},
														{-4 * 0.025, -1 * 0.025, 0}
													}
		local final_offset = {0.3, -0.3, 0.0125}

		-- final block location
		local final_block_loc = {}
		for i=1,7 do
			local tmp = {final_block_rel_loc[block_id_map[i]][1] + final_offset[1],
									 final_block_rel_loc[block_id_map[i]][2] + final_offset[2],
									 final_block_rel_loc[block_id_map[i]][3] + final_offset[3]}
			table.insert(final_block_loc, tmp)
		end

    -- block catch order in BLOCK ID (SIM)
    local order = {7, 4, 6, 1, 3, 5, 2}



    -- heuristic values
    --local xoffset = 0.05 -- for set gripper appropriately
    --local block_xoffset = {-0.065, -0.017, -0.02, -0.03, -0.075, -0.075, -0.05}
    local z_max_offset = 0.3
		local z_1_offset = 0.25
		local z_2_offset = 0.2
		local z_3_offset = 0.1

    local z_min_offset = 0.05
		local z_put_offset = 0.3--0.05
    local gripInit = {0.06, 0.06}
    local gripHold = {0.02, 0.02}
    local gripRelease = gripInit


    -- assemble loc info
    --local loc_offset = {0.4, -0.4, 0.0125}
    --local block_loc = {{-0.025, 0.075, 0.0},
    --                   {-0.025, 0.1, 0.1},
    --                   {-0.075, 0.05, 0.0},
    --                   {-0.025, 0.0375, 0.0},
    --                   {-0.1, 0.1, -0.006},
    --                   {-0.1, 0.0, 0.0},
    --                   {-0.025, 0.0, 0.0}}

		--local block_hold_pos_offset = {{}}


    function wait()
        os.execute("sleep 4")
    end

    --function loc_cal(idx, x) -- x: 0=x, 1=y
    --    return loc_offset[x] + block_loc[idx][x]
    --end

    print("block xpos", block_xpos)
    print("block ypos", block_ypos)

    function pickBlock(idx)
        hcm.set_arm_grabxyz({refine_cubes_xpos[idx], block_ypos[idx], z_max_offset})
        arm_ch:send'moveto'
        wait()

        hcm.set_arm_grabxyz({refine_cubes_xpos[idx], block_ypos[idx], z_min_offset})
        arm_ch:send'moveto'
        wait()

        hcm.set_arm_gripperTarget(gripHold)
        wait()

        hcm.set_arm_grabxyz({refine_cubes_xpos[idx], block_ypos[idx], z_max_offset})
        arm_ch:send'moveto'
        wait()

        hcm.set_arm_grabxyz({final_block_loc[idx][1], final_block_loc[idx][2], z_max_offset})
        arm_ch:send'moveto'
        wait()

				--hcm.set_arm_grabxyz({final_block_loc[idx][1], final_block_loc[idx][2], z_1_offset + 0.01})
        --arm_ch:send'moveto'
        --wait()

				--hcm.set_arm_grabxyz({final_block_loc[idx][1], final_block_loc[idx][2], z_2_offset + 0.01})
        --arm_ch:send'moveto'
        --wait()

				--hcm.set_arm_grabxyz({final_block_loc[idx][1], final_block_loc[idx][2], z_3_offset + 0.01})
        --arm_ch:send'moveto'
        --wait()

        hcm.set_arm_grabxyz({final_block_loc[idx][1], final_block_loc[idx][2], z_put_offset + 0.01})
        arm_ch:send'moveto'
        wait()
				gt = hcm.get_arm_gripperTarget()

				print(gt[1])
				gt[1] = gt[1] + 0.06
				gt[2] = gt[2] + 0.006
				print(gt[2])

        hcm.set_arm_gripperTarget(gt)
        wait()

        hcm.set_arm_grabxyz({final_block_loc[idx][1], final_block_loc[idx][2], z_max_offset})
        arm_ch:send'moveto'
        wait()

				hcm.set_arm_gripperTarget(gripInit)
        wait()
    end

    hcm.set_arm_gripperTarget(gripInit)
    wait()

    --for i=1,7 do
    --    pickBlock(order[i])
    --end
		--pickBlock(3)




	-- Interactive LuaJIT
	package.path = package.path..';'..HOME..'/Tools/iluajit/?.lua'
	dofile(HOME..'/Tools/iluajit/iluajit.lua')
end
