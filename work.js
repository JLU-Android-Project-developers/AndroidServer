var express = require('express')//引入express模块
var multer = require('multer')
var cors = require('cors')
var bodyParser = require('body-parser')
var algorithm = require('./algorithm.js')
var path = require('path')

var app = express()
var storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, '/home/guest/Android-Smoke-app/img_in')
    },
    filename: function (req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now()+'.jpg')
    }
})
var upload = multer({ storage:storage })

app.use(cors())
app.use(bodyParser.json())
app.use(bodyParser.raw())
app.use(bodyParser.urlencoded())

//创建get请求服务，路由为默认路径
app.get('/',(req,res)=>{
    console.log('Node.js received a request')
    res.end('Hello world from Node.js!')
})

//app.use('/img_out/img_out',express.static('./img_out/img_out'))

app.post("/upload", upload.single('file'), function (req, res) {    

    console.log(req.file)
    let img_path= req.file.path
    algorithm.start(img_path,data =>{
        let msg = {
            code:200,
            message:'ok',
            url:"test.jpg",
            data:data
        }
        console.log(msg)
        res.send(msg)
        return;
    })
})
var jsonParser = bodyParser.json();
app.post("/download",function (req, res) {

    var tmp = req.body;
    console.log(tmp.downloadPath);
    res.download(tmp.downloadPath, 'result.jpg');
    
})

//设置服务端监听端口为8089，成功后回调在控制台上打印提示
let server = app.listen(8090,()=>{
    console.log('The server is listening on port : 8090')
})

