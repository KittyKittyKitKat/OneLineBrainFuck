((sys:=__import__('sys'),re:=__import__('re'),Enum:=__import__('enum').Enum,auto:=__import__('enum').auto,signal:=__import__('signal'),INFINTIY:=float('inf'),NEGATIVE_INFINITY:=float('-inf'),EOF_BEHAVIOUR:=Enum('EOF_BEHAVIOUR',{'ZERO':auto(),'NO_CHANGE':auto()}),throw:=lambda exception,msg='':(_ for _ in()).throw(exception(msg)),signal.signal(signal.SIGINT,lambda *_:(print(),sys.exit(0))),BrainfuckInterpreter:=type('BrainfuckInterpreter',(object,),{'__init__':lambda self,code,*,bits=8,unsigned=True,bit_wrapping=True,tape_wrapping=False,max_tape_size=32768,eof_behavior=EOF_BEHAVIOUR.NO_CHANGE,extended_unicode_support=True:(None,throw(ValueError,'Invalid EOF behaviour specified')if eof_behavior not in EOF_BEHAVIOUR.__members__.values()else None,setattr(self,'code',BrainfuckInterpreter.strip_code(code)),setattr(self,'bracket_matches',self.bracket_balance_match()),setattr(self,'reversed_bracket_matches',{value:key for key,value in self.bracket_matches.items()}),(setattr(self,'max_cell_value',2**bits-1),setattr(self,'min_cell_value',0))if unsigned else(setattr(self,'max_cell_value',2**bits//2-1),setattr(self,'min_cell_value',~self.max_cell_value)),setattr(self,'bit_wrapping',bit_wrapping),setattr(self,'max_tape_size',max_tape_size),setattr(self,'tape_wrapping',False if self.max_tape_size==INFINTIY else tape_wrapping),setattr(self,'eof_behavior',eof_behavior),setattr(self,'extended_unicode_support',extended_unicode_support),setattr(self,'tape',[0]*(self.max_tape_size if self.tape_wrapping else 1)),setattr(self,'pointer',0),setattr(self,'program_counter',0),setattr(self,'stdout_stream',''),setattr(self,'command_map',{'>':self.move_right,'<':self.move_left,'+':self.increment,'-':self.decrement,'.':self.write,',':self.read,'[':self.jump_if_zero,']':self.jump_unless_zero}))[0],'from_file':classmethod(lambda cls,code_file,**kwargs:(fp:=open(code_file),code:=''.join([line.strip()for line in fp]),fp.close(),cls(code,**kwargs))[-1]),'strip_code':staticmethod(lambda raw_code,allowed_chrs='+-<>,.[]':re.sub(r'[^'+re.escape(allowed_chrs)+']','',raw_code.strip())),'check_pointer':lambda self:(None,throw(IndexError,f'Cannot access index {self.pointer} on tape')if self.pointer<0 and not self.tape_wrapping else None)[0],'bracket_balance_match':lambda self:(bracket_matches:={},opening_positions:=[],bracket_queue:=[],[((bracket_queue.append(']'),opening_positions.append(x))if i=='['else((throw(SyntaxError,'Mismatched brackets')if not bracket_queue or i != bracket_queue.pop()else None,bracket_matches.__setitem__(opening_positions.pop(),x))if i==']' else None))for x,i in enumerate(self.code)],throw(SyntaxError,'Mismatched brackets')if bracket_queue else None,)[0],'move_right':lambda self:(None,setattr(self,'pointer',self.pointer+1),(self.tape.append(0)if len(self.tape)<self.max_tape_size else throw(MemoryError,'Maximum tape length of {self.max_tape_size} exceeded'))if self.pointer>=len(self.tape)else None)[0],'move_left':lambda self:(None,setattr(self,'pointer',self.pointer-1))[0],'increment':lambda self:(None,self.check_pointer(),(self.tape.__setitem__(self.pointer,self.min_cell_value)if self.bit_wrapping else throw(ValueError,f'Minimum cell value of {self.min_cell_value} exceeded'))if self.tape[self.pointer]==self.max_cell_value else self.tape.__setitem__(self.pointer, self.tape[self.pointer]+1))[0],'decrement':lambda self:(None,self.check_pointer(),(self.tape.__setitem__(self.pointer,self.max_cell_value)if self.bit_wrapping else throw(ValueError,f'Maximum cell value of {self.max_cell_value} exceeded'))if self.tape[self.pointer]==self.min_cell_value else self.tape.__setitem__(self.pointer,self.tape[self.pointer]-1))[0],'write':lambda self:(None,self.check_pointer(),c:=[self.tape[self.pointer]],UTF8:=[True],(sentinel:=object(),n:=[1],v:=[c[0]&0x3f],h:=[0xc0],cc:=[ord(self.stdout_stream[len(self.stdout_stream)-n[0]])or 0],*iter(lambda:((sentinel,UTF8.__setitem__(0,False))[0]if cc[0]>0xff or not(cc[0]and 0x80)else((sentinel,c.__setitem__(0,v[0]|(cc[0]&~h[0])<<(n[0]*6)),setattr(self,'stdout_stream',self.stdout_stream[:len(self.stdout_stream)-n[0]]))[0]if(cc[0]&h[0])==h[0]and not(cc[0]&((h[0]>>1)&(~h[0])))else(v.__setitem__(0,v[0]|((cc[0]&0x3f)<<(n[0]*6)))if cc[0]&0x80 and not(cc[0]&0x40)and n[0]<5 else None,h.__setitem__(0,h[0]|(h[0]>>1)),n.__setitem__(0,n[0]+1),cc.__setitem__(0,(ord(self.stdout_stream[len(self.stdout_stream)-n[0]])or 0))))),sentinel))if self.extended_unicode_support and c[0]>0x7f and len(self.stdout_stream)else None,setattr(self,'stdout_stream',self.stdout_stream+chr(c[0])),print(chr(c[0]),end='')if UTF8[0] else None)[0],'read':lambda self:(None,self.check_pointer(),sys.stdout.flush(),val:=sys.stdin.readline(),val:=ord(val[0])if val else(0 if self.eof_behavior is EOF_BEHAVIOUR.ZERO else None),self.tape.__setitem__(self.pointer,val)if val is not None else None)[0],'jump_if_zero':lambda self:(None,setattr(self,'program_counter',self.bracket_matches[self.program_counter])if not self.tape[self.pointer]else None)[0],'jump_unless_zero':lambda self:(None,setattr(self,'program_counter',self.reversed_bracket_matches[self.program_counter])if self.tape[self.pointer]else None)[0],'run_program':lambda self:(None,program_len:=len(self.code),(sentinel:=object(),*iter(lambda:(self.command_map[self.code[self.program_counter]](),setattr(self,'program_counter',self.program_counter+1))if self.program_counter<program_len else sentinel,sentinel)))}),(bfi:=BrainfuckInterpreter.from_file('Programs/15-puzzle.bf'),bfi.run_program()))if __name__=='__main__'else None)