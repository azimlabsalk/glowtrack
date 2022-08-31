/*

#define PORT_SH_CLK_UV PORTD
#define BIT_SH_CLK_UV 2

#define PORT_SH_DIN_UV PORTD
#define BIT_SH_DIN_UV 3

#define PORT_SH_CLK_VIS PORTB
#define BIT_SH_CLK_VIS 0

#define PORT_SH_DIN_VIS PORTD
#define BIT_SH_DIN_VIS 4

//direct port manipulation for quick toggling of shift register data
inline void SH_CLK_UV_HIGH(){  PORT_SH_CLK_UV |= (0B00000001 << BIT_SH_CLK_UV);}
inline void SH_CLK_UV_LOW(){  PORT_SH_CLK_UV &= ~(0B00000001 << BIT_SH_CLK_UV);}

inline void SH_DIN_UV_HIGH(){  PORT_SH_DIN_UV |= (0B00000001 << BIT_SH_DIN_UV);}
inline void SH_DIN_UV_LOW(){  PORT_SH_DIN_UV &= ~(0B00000001 << BIT_SH_DIN_UV);}

inline void SH_CLK_VIS_HIGH(){  PORT_SH_CLK_VIS |= (0B00000001 << BIT_SH_CLK_VIS);}
inline void SH_CLK_VIS_LOW(){  PORT_SH_CLK_VIS &= ~(0B00000001 << BIT_SH_CLK_VIS);}

inline void SH_DIN_VIS_HIGH(){  PORT_SH_DIN_VIS |= (0B00000001 << BIT_SH_DIN_VIS);}
inline void SH_DIN_VIS_LOW(){  PORT_SH_DIN_VIS &= ~(0B00000001 << BIT_SH_DIN_VIS);}


//shifts out one byte of data onto the UV register chain. Line is shared with the camera. 
void program_UV_reg(byte data){
  byte i;
  for(i=0; i<8; i++){
    //set data bit
    if(data & 0B00000001)
      SH_DIN_UV_HIGH();
    else
      SH_DIN_UV_LOW();
      
    //pulse clock
    SH_CLK_UV_HIGH();
    SH_CLK_UV_LOW();
  
    //shift data
    data = data >> 1;
  }
}

//shifts out one byte of data onto the VIS register chain. 
void program_VIS_reg(byte data){
  byte i;
  for(i=0; i<8; i++){
    //set data bit
    if(data & 0B00000001)
      SH_DIN_VIS_HIGH();
    else
      SH_DIN_VIS_LOW();
      
    //pulse clock
    SH_CLK_VIS_HIGH();
    SH_CLK_VIS_LOW();
    
    //shift data
    data = data >> 1;
  }
}

//shifts out one byte of data into the camera. This is the first register on the UV register chain. 
inline void program_camera_reg(byte data){
  program_UV_reg(data);
}

*/
