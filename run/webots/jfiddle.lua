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

    -- block catch order
    local order = {7, 4, 6, 1, 3, 5, 2}

    -- heuristic values
    local xoffset = 0.05 -- for set gripper appropriately
    local block_xoffset = {-0.065, -0.017, -0.02, -0.03, -0.075, -0.075, -0.05}
    local z_max_offset = 0.2
    local z_min_offset = 0.013
    local gripInit = {0.04, 0.04}
    local gripHold = {0.01, 0.01}
    local gripRelease = gripInit

    -- assemble loc info
    local loc_offset = {0.6, -0.2, 0.0125}
    local block_loc = {{-0.025, 0.075, 0.0},
                       {-0.025, 0.1, 0.1},
                       {-0.075, 0.05, 0.0},
                       {-0.025, 0.0375, 0.0},
                       {-0.1, 0.1, -0.006},
                       {-0.1, 0.0, 0.0},
                       {-0.025, 0.0, 0.0}}


    function wait()
        os.execute("sleep 3")
    end

    function loc_cal(idx, x) -- x: 0=x, 1=y
        return loc_offset[x] + block_loc[idx][x]
    end

    print("block xpos", block_xpos)
    print("block ypos", block_ypos)

    function pickBlock(idx)
        hcm.set_arm_grabxyz({block_xpos[idx] + block_xoffset[idx], block_ypos[idx], z_max_offset})
        arm_ch:send'moveto'
        wait()

        hcm.set_arm_grabxyz({block_xpos[idx] + block_xoffset[idx], block_ypos[idx], z_min_offset})
        arm_ch:send'moveto'
        wait()

        hcm.set_arm_gripperTarget(gripHold)
        wait()

        hcm.set_arm_grabxyz({block_xpos[idx] + block_xoffset[idx], block_ypos[idx], z_max_offset})
        arm_ch:send'moveto'
        wait()

        hcm.set_arm_grabxyz({loc_cal(idx, 1), loc_cal(idx, 2), z_max_offset})
        arm_ch:send'moveto'
        wait()

        hcm.set_arm_grabxyz({loc_cal(idx, 1), loc_cal(idx, 2), z_min_offset + block_loc[idx][2]})
        arm_ch:send'moveto'
        wait()

        hcm.set_arm_gripperTarget(gripRelease)
        wait()

        hcm.set_arm_grabxyz({loc_cal(idx, 1), loc_cal(idx, 2), z_max_offset})
        arm_ch:send'moveto'
        wait()
    end

    hcm.set_arm_gripperTarget(gripInit)
    wait()

    for i=1,7 do
        pickBlock(order[i])
    end




    
	-- Interactive LuaJIT
	package.path = package.path..';'..HOME..'/Tools/iluajit/?.lua'
	dofile(HOME..'/Tools/iluajit/iluajit.lua')
end

