_pt = """
  a + 2e-  B-d |-a  +-aA +C e--A7--|  --|+--   -7  --m  +
  -|  - ---4 + -b --Tt- |R-- |P-1 |--  |-   -  s-  S-- |
-  |-e  +-pA-+N ---B
--|  --| --   -a  +Cu  |-- -|  -  --a + Nc--|Cg-  A-- |P-A |-   |- B -  r-e  -n |--  |-e  +-mA-+S  --E---|- - | --   -   +L  -|
-- |- - ---i + 3h--|Fb-  I-- |R-  |-   +A i--   -| A-  |--  +-d -+-k -+G  --M --|  -   -s  |
-- +T  -| -- |- a --
r +-5g-- No-  S-  |--  |-   -  S-t  --| --  |--  --m  +-s--+D- -|L --|  -   -A  +S --|R- -|--- |- B --Be +|7n-- Zn- | -  |-c  +
 B -T r-|  --| --  +--  --d  +-d -+E  -|  - | -5-- -a  +Z --|I-  |--2 |--B -- i + 8b-1 G-- |
-  |-   +h N-- 4--| 
--||--   -- ---y- +Fw -|Y  --  -N  -r -- e- |M---|A-  |--e |- B -- n + 1I-  S-t |-   |-h  +L F--P6--| ---||-- 8 --  --r  +A  -|
 - -- - | -r --Sr- |R-- |T-- |--  |- B -A e-| A -  K-  |-a  +-UA-+P  --CH--|  --|--- - -r  +
b  +-  -|  -  --K + 
o--+ u-  P-- |B-H |--  |- B -  e-i  -  |--  |-a  +-uA-+P ---C---|A --| --   -C  +B  -|-- -|- -  --c + Mu--|Ml-- C-- |A-  |-   +
 l -  r-| C-n |--  +-r  +-me-+E  --F --|- - | -6   --  +H  -| -- |- 3 ---V + 4d--|Ci- |S-- |
-  |-   +n P-   --| 
-  |--  --m -+-f--+T
 --N --|  -T  -a  +R-- | - -| -- |- g --Ln +-6d-- Ct-  T-  |-   |-   -A l-i  --| --  |--  --u  +-m -+H  -|
 - |  --- -f  +  --|O-  |--- |- B -- o +| n-- G -  X-  |-i  +L C--P3 -| N--| --  |-- ---b  +
o -+T  -|  -   -b---
W- +N --|P-  |--i |-
B -- u +  e-  A-  |--  |-c  +d O--N5--| ---||--   --  --o  +R  -|L  --- -   -Y --Cs- |T---|H-  |--  |-2  -
""".strip('\n')
periodic_table = ''.join(_pt[i*197%1473] for i in range(1473))