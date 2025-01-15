import GlobalVars as GV
import sys
from collections import deque
from Energy import op_count

cdef class Switch:
    def __init__(self, int x, int y):

        # Activation flags
        self.is_active = False
        self.flag_deact = False
        self.in_activation_list = False
        self.in_deactivation_list = False

        # Switch position
        self.x = x
        self.y = y

        # Connected router
        self.router = None

        # Chip edge
        self.is_edge = [False for _ in range(<int>DirectionIndex.direction_max)] # LEFT, BOT, TOP, RIGHT

        if self.x == 0: self.is_edge[<int>DirectionIndex.left] = True
        if self.y == 0: self.is_edge[<int>DirectionIndex.bot] = True
        if self.y == GV.sim_params['max_core_y']: self.is_edge[<int>DirectionIndex.top] = True
        if self.x == GV.sim_params['max_core_x'] - 1: self.is_edge[<int>DirectionIndex.right] = True

        # 4 switches around
        self.switches = [None, None, None, None]

        # Switch delay
        self.delay = [0, 0, 0, 0, 1]
        self.delay[<int>DirectionIndex.left] = 1
        self.delay[<int>DirectionIndex.bot] = 1
        self.delay[<int>DirectionIndex.top] = 1
        self.delay[<int>DirectionIndex.right] = 1

        # Input data
        self.i_data = [None, None, None, None, None]

        # Output data
        self.o_data = [None, None, None, None, None]
        self.o_data_next = [None, None, None, None, None]

        # Output buffer
        self.o_buf = [deque(), deque(), deque(), deque(), deque()]
        self.buf_pos = [0, 0, 0, 0, 0]

    cpdef put_buf(self, pck):
        cdef int dx
        cdef int dy
        cdef int direction

        if not pck:
            return

        dx = pck['dst_x'] - self.x
        dy = pck['dst_y'] - self.y
        # Prioritize y diretion first (to enable external-to-core routing)
        if dy < 0:
            direction = <int>DirectionIndex.bot
        elif dy > 0:
            direction = <int>DirectionIndex.top
        elif dx < 0:
            direction = <int>DirectionIndex.left
        elif dx > 0:
            direction = <int>DirectionIndex.right
        else: # dx, dy = 0
            direction = 4

        # The o_buf should be sth like:
        # buffer[(), (), ... ()] => number of entries should be at least delay size
        # 1) if the current number of entries is smaller than the delay size =>
        #    append empty entries until the buffer size equals to the delay size
        # 2) if the current number of entries is larger than the delay size =>
        #    append the packet to the last entry
        # 3) if the number of packets inside the entry exceeds the bandwidth
        #    generate a new entry
        while len(self.o_buf[direction]) < self.delay[direction]:
            if len(self.o_buf[direction]) == self.delay[direction] - 1:
                self.o_buf[direction].append([None for _ in range(GV.timing_params["sw_to_sw_bw"])])
                self.buf_pos[direction] = 0
            else:
                self.o_buf[direction].append(None)

        # Set the buffer capacity
        # Append the packet to the last entry and increment the buffer pos (within the entry)
        # If the number of packets exceeds the buffer size => add a new entry
        if self.buf_pos[direction] == GV.timing_params['sw_to_sw_bw']:
            self.o_buf[direction].append([None for _ in range(GV.timing_params['sw_to_sw_bw'])])
            self.buf_pos[direction] = 0
        # We append the packet for the first entry
        self.o_buf[direction][-1][self.buf_pos[direction]] = pck
        self.buf_pos[direction] += 1

        # 
        GV.max_buffer_size = max(GV.max_buffer_size, len(self.o_buf[direction]))


    cpdef put_activation_list(self, switch, activation_list):
        if not switch.in_activation_list:
            switch.in_activation_list = True
            activation_list.append(switch)


    cpdef put_deactivation_list(self, switch, deactivation_list):
        if not switch.in_deactivation_list:
            switch.in_deactivation_list = True
            deactivation_list.append(switch)

    cpdef advance(self, activation_list):
        cdef int direction
        cdef dict pck
        cdef int pos

        for direction in range(4):
            self.o_data[direction] = self.o_data_next[direction]

        if self.o_data_next[4]:
            assert self.router
            for pck in self.o_data_next[4]:
                if pck:
                    self.router.recv_buf.append(pck)
                    op_count['recv_buf_push'] += 1

        for direction in range(4):
            if self.o_data[direction] and not self.is_edge[direction]:
                self.put_activation_list(self.switches[direction], activation_list)


    cpdef calc_next(self, deactivation_list):
        for direction in range(4):
            if not self.is_edge[direction]:
                # prepare i_data from the connected switches' o_data
                self.i_data[direction] = self.switches[direction].o_data[3-direction]
                # reset o_data of the connected switches
                self.switches[direction].o_data[3-direction] = None

        # Then, we should put the input data to the o_buf
        for direction in range(5):
            if self.i_data[direction]:
                for pck in self.i_data[direction]:
                    self.put_buf(pck)
            self.i_data[direction] = None

        # If all of the output buffer is empty => deactivate the switch
        self.flag_deact = True

        # Lastly, we set the next output data by popping the o_buf entries
        for direction in range(5):
            if self.o_buf[direction]:
                # If any of the buffer is not empty => 
                # do not deactivate the switch
                self.flag_deact = False
                self.o_data_next[direction] = self.o_buf[direction].popleft()
                op_count['NoC_hop'] += 1
            else:
                self.o_data_next[direction] = None

        #self.flag_deact = False
        if self.flag_deact:
            self.put_deactivation_list(self, deactivation_list)


cdef class NoC:
    def __init__(self):

        self.max_x = GV.sim_params['max_core_x']
        self.max_y = GV.sim_params['max_core_y'] + 1

        # Initialize switch
        self.sw = [[Switch(x, y) for y in range(self.max_y)] for x in range(self.max_x)]

        # Switch connetion
        for x in range(self.max_x):
            for y in range(self.max_y):
                left_x = x - 1
                right_x = x + 1
                bot_y = y - 1
                top_y = y + 1

                if left_x == -1: left_x = self.max_x - 1
                if right_x == self.max_x: right_x = 0
                if bot_y == -1: bot_y = self.max_y - 1
                if top_y == self.max_y: top_y = 0

                self.sw[x][y].switches[<int>DirectionIndex.left] = self.sw[left_x][y]
                self.sw[x][y].switches[<int>DirectionIndex.bot] = self.sw[x][bot_y]
                self.sw[x][y].switches[<int>DirectionIndex.top] = self.sw[x][top_y]
                self.sw[x][y].switches[<int>DirectionIndex.right] = self.sw[right_x][y]

        # Active switch list
        self.active_sw_list = []
        #self.active_sw_list.append(self.merge_split)
        #self.active_sw_list.append(self.merge_split_external)


    cpdef noc_advance(self):
        cdef list activation_list = []
        cdef list deactivation_list = []
        cdef object switch

        if len(self.active_sw_list) > 0:
            GV.valid_advance = True

        # Switch advance
        for switch in self.active_sw_list:
            switch.advance(activation_list)

        # Activate receiving switches
        for switch in activation_list:
            switch.in_activation_list = False
            # The switches that receive the packet
            # should perform (calc_next and perform advance @ next timestep)
            self.activate_sw(switch)
        
        # Send spike messages to switch
        for core in GV.cores:
            self.send_pck_to_sw(core.router)
        
        # Switch calc_next
        for switch in self.active_sw_list:
            switch.calc_next(deactivation_list)
        
        # Deactivate empty switch
        for switch in deactivation_list:
            switch.in_deactivation_list = False
            self.deactivate_sw(switch)
        
        if len(self.active_sw_list) > 0:
            GV.valid_advance = True

    cpdef send_pck_to_sw(self, router):
        if not router.send_buf: return

        packet = [None for _ in range(GV.timing_params['core_to_sw_bw'])]

        for i in range(GV.timing_params['core_to_sw_bw']):
            top_packet = router.send_buf[0]
            if top_packet["cyc"] <= GV.cyc:
                packet[i] = router.send_buf.pop(0)["packet"]

        router.sw.i_data[4] = packet
        self.activate_sw(router.sw)


    cpdef activate_sw(self, switch):
        if not switch.is_active:
            switch.is_active = True
            self.active_sw_list.append(switch)


    cpdef deactivate_sw(self, switch):
        if switch.is_active:
            switch.is_active = False
            self.active_sw_list.remove(switch)
