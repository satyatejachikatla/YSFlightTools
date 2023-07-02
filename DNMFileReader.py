import numpy as np
import cv2
import os

class Vertex:
    def __init__(self,id,x,y,z):
        self.id = id
        self.vertices = np.array([x,y,z])

    def x(self):
        return self.vertices[0]
    def y(self):
        return self.vertices[1]
    def z(self):
        return self.vertices[2]
    
    def getObjString(self):
        return f'v {self.x()} {self.y()} {self.z()}\n'

class Normal:
    def __init__(self,id,x,y,z):
        self.id = id
        self.normal = np.array([x,y,z])
        norm = np.linalg.norm(self.normal)
        if norm != 0:
            self.normal = self.normal/norm

    def x(self):
        return self.normal[0]
    def y(self):
        return self.normal[1]
    def z(self):
        return self.normal[2]
    
    def getObjString(self):
        return f'vn {self.x()} {self.y()} {self.z()}\n'

class Texture:
    def __init__(self):

        # Meta
        self.name = None
        self.textureImgFileName = None
        self.mtlFileName = None

        self.savePath = None
        self.textureImgSavePath = None
        self.mtlSavePath = None

        # Data
        self.colorMap = []
    
    def addNewColor(self, color):
        self.colorMap.append(color)
        return len(self.colorMap) - 1
    
    def addSaveLocation(self, name, filepath):
        self.name = name
        self.textureImgFileName = self.name + '.png'
        self.mtlFileName = self.name + '.mtl'

        self.savePath = filepath
        self.textureImgSavePath = f'{self.savePath}/{self.textureImgFileName}'
        self.mtlSavePath = f'{self.savePath}/{self.mtlFileName}'

    def saveTexture(self):
        # Repeat factor increase the number of pixels per index to avoid bad color read in obj material
        REPEAT_FACTOR = 20
        REPEAT_FACTOR_BY_2 = REPEAT_FACTOR//2
        # import pdb;pdb.set_trace()

        if self.textureImgSavePath:
            colorMap = np.array([self.colorMap]).astype(np.uint8)
            try:
                b = np.repeat(colorMap[:,:,0],REPEAT_FACTOR)
                g = np.repeat(colorMap[:,:,1],REPEAT_FACTOR)
                r = np.repeat(colorMap[:,:,2],REPEAT_FACTOR)

                b = np.hstack([b[REPEAT_FACTOR_BY_2:],b[:REPEAT_FACTOR_BY_2]])
                g = np.hstack([g[REPEAT_FACTOR_BY_2:],g[:REPEAT_FACTOR_BY_2]])
                r = np.hstack([r[REPEAT_FACTOR_BY_2:],r[:REPEAT_FACTOR_BY_2]])

                colorMap = cv2.merge((r,g,b))
                colorMap = colorMap.astype(np.uint8).reshape((1,-1,3))

                cv2.imwrite(self.textureImgSavePath,colorMap)
            except:
                open(self.textureImgSavePath,'w').close()
        else:
            raise Exception('Failed to save')

    def getMtlString(self):
        mtlStr = ''
        if self.textureImgSavePath:
            mtlStr += f'newmtl {self.name}\n'
            mtlStr += f'Ka 1.000000 1.000000 1.000000\n'
            mtlStr += f'Kd 1.000000 1.000000 1.000000\n'
            mtlStr += f'Ks 0.000000 0.000000 0.000000\n'
            mtlStr += f'Tr 1.000000\n'
            mtlStr += f'illum 1\n'
            mtlStr += f'Ns 0.000000\n'
            mtlStr += f'map_Kd {self.textureImgSavePath}\n'
        else:
            raise Exception('Failed to generate string')
        return mtlStr
    
    def saveMtl(self):
        fp = open(self.mtlSavePath,'w')
        fp.write(self.getMtlString())
        fp.close()

    def getObjStringMtllib(self):
        return f'mtllib {self.mtlFileName}\n'
    
    def getObjStringUsemtl(self):
        return f'usemtl {self.name}\n'
    
    def getObjStringVt(self):
        objStr = ''
        totalCId = len(self.colorMap)
        for cId, c in enumerate(self.colorMap):
            objStr += f'vt {cId/totalCId}\n'
        return objStr

class Face:
    def __init__(self):
        self.bright = False
        self.vertices = []
        self.normal = None
        self.color = None
    
    def getObjString(self,vertexOffset,textureOffset,normalOffset):
        vertexListForObj = [ f'{vertexOffset+v+1}/{textureOffset+self.color+1}/{normalOffset+self.normal+1}' for v in self.vertices]
        objStr = 'f ' + ' '.join(map(str,vertexListForObj)) + '\n'
        return objStr

class Model:
    def __init__(self):
        self.name = None
        self.vertices = []
        self.normals = []
        self.faces = []
        self.texture = Texture()
    
    def getObjString(self,vertexOffset,textureOffset,normalOffset):
        objStr = f's 1\n' # For smooth shading by default
        objStr += f'o {self.name}\n'
        
        objStr += self.texture.getObjStringMtllib()

        for v in self.vertices:
            objStr += v.getObjString()
        
        objStr += self.texture.getObjStringVt()

        for n in self.normals:
            objStr += n.getObjString()
        
        objStr += self.texture.getObjStringUsemtl()

        for f in self.faces:
            objStr += f.getObjString(vertexOffset,textureOffset,normalOffset)
        return objStr
    
    def getOffsets(self):
        return len(self.vertices),len(self.texture.colorMap),len(self.normals)

class DNMFile:
    def __init__(self,name,filepath,saveFolder):

        #Meta properties
        self.name = name
        self.nameObj = name + '.obj'
        self.filepath = filepath
        self.cls = __class__

        #Data
        self.version = None
        self.subModels = []

        self._readFile()
        self._writeObj(saveFolder)
    
    def _readFile(self):
        fp = open(self.filepath,'r')
        fp_lines = fp.readlines()

        # State info
        isPCKHitBeforeSURF = False
        isFaceRead = False

        for l in fp_lines:
            l = l.strip()
            try:
                parserTag = l.split()[0]
            except IndexError:
                continue
            
            if parserTag == 'V' or parserTag == 'VER':
                if (isFaceRead):
                    vertexData = l.split()
                    for v in vertexData[1:]:
                        try:
                            self.subModels[-1].faces[-1].vertices.append(int(v))
                        except:
                            raise Exception(f'Failed to Read Vertex Info, unknown tag {v}')
                    
                else:
                    vertexData = l.split()
                    try:
                        vertexId = len(self.subModels[-1].vertices)
                        x = float(vertexData[1])
                        y = float(vertexData[2])
                        z = float(vertexData[3])
                        self.subModels[-1].vertices.append(Vertex(vertexId,x,y,z))
                    except ValueError:
                        raise Exception(f'Failed to Read Vertex Info, unknown tag {v}')
                    
                    try:
                        if vertexData[4] == 'R':
                            self.subModels[-1].vertices[-1] = np.round(self.subModels[-1].vertices[-1],decimals=2)
                    except:
                        pass

            elif parserTag == 'C' or parserTag == 'COL':
                colorData = l.split()
                try:
                    color = [int(colorData[1]),int(colorData[2]),int(colorData[3])]
                    colorId = self.subModels[-1].texture.addNewColor(color)
                    self.subModels[-1].faces[-1].color = colorId
                except:
                    raise Exception(f'Failed to Read Face Color')

            elif parserTag == 'F' or parserTag == 'FAC':
                isFaceRead = True
                self.subModels[-1].faces.append(Face())
            
            elif parserTag == 'B':
                self.subModels[-1].faces[-1].bright = True

            elif parserTag == 'N' or parserTag == 'NOR':
                normalData = l.split()
                try:
                    normalId = len(self.subModels[-1].normals)
                    #check what is 1 , 2 , 3
                    x = float(normalData[4])
                    y = float(normalData[5])
                    z = float(normalData[6])
                    self.subModels[-1].normals.append(Normal(normalId,x,y,z))
                    self.subModels[-1].faces[-1].normal = normalId
                except:
                    raise Exception(f'Failed to Read Face Normal')

            elif parserTag == 'E' or parserTag == 'END':
                isFaceRead = False

            elif parserTag == 'PCK':
                subModelData = l.split()
                self.subModels.append(Model())
                try:
                    self.subModels[-1].name = subModelData[1]
                except:
                    raise Exception(f'Failed to Read Sub Model Info, {subModelData}')
                isPCKHitBeforeSURF = True

            elif parserTag == 'SURF':
                if not isPCKHitBeforeSURF:
                    self.subModels.append(Model())
                    try:
                        self.subModels[-1].name = self.name
                    except:
                        raise Exception(f'Failed to Read Sub Model Info')
                else:
                    isPCKHitBeforeSURF = False

            elif parserTag == 'DYNAMODEL':
                pass

            elif parserTag == 'DNMVER':
                parserTag, version = l.split()
                self.version = int(version)
            else:
                pass
                # raise Exception(f'Unknown tag {parserTag}')
    
    def _writeObj(self,filepath):

        objSaveName = f'{filepath}/{self.nameObj}'
        mtlSavePath = f'{filepath}'

        fp = open(objSaveName, 'w')

        vertexOffset,textureOffset,normalOffset = 0,0,0

        for model in self.subModels:

            model.texture.addSaveLocation(model.name,mtlSavePath)
            model.texture.saveTexture()
            model.texture.saveMtl()

            fp.write(model.getObjString(vertexOffset,textureOffset,normalOffset))
            fp.write('\n')

            _vertexOffset,_textureOffset,_normalOffset = model.getOffsets()
            vertexOffset,textureOffset,normalOffset = vertexOffset+_vertexOffset, textureOffset+_textureOffset, normalOffset+_normalOffset
        fp.close()


if __name__ == '__main__':
    dnm = DNMFile('foo',r'D:\Install\Ysflight\YSFLIGHT\aircraft\t2blue.dnm','./modelFolder')
    # dnm = DNMFile('foo',r'D:\Install\Ysflight\YSFLIGHT\aircraft\f22.dnm','./modelFolder')

    # dnm = DNMFile('foo',r'./temp.srf','./modelFolder')
    print(dnm.version)