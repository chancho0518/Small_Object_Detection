import { Component, ElementRef, OnInit, Renderer2, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage implements OnInit{
  @ViewChild('video', { static: false }) videoElement: ElementRef;
  @ViewChild('canvas', { static: false }) canvas: ElementRef;

  videoBoundingRect;
  videoWidth: number = 0;
  videoHeight: number = 0;
  constraints = {
    video: {
      facingMode: "environment"
    }
  };
  model: string = 'yolo';

  showVideo: boolean = false;
  svgEnabled: boolean = true;
  doSpinner:  boolean = false;
  apiUrl = 'http://127.0.0.1:5000';

  currentDetections;

  constructor(private renderer: Renderer2, private http: HttpClient) {}

  ngOnInit() {
    this.startCamera()
  }

  async startCamera() {
    if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
      (navigator.mediaDevices.getUserMedia(this.constraints))
      .then(this.attachVideo.bind(this))
      .catch(this.handleError);
      this.showVideo = true;
      this.svgEnabled = false;
    } else {
      alert('Sorry, camera is not available!!');
    }
  }

  attachVideo(stream) {
    this.renderer.setProperty(this.videoElement.nativeElement, 'srcObject', stream);
    this.renderer.listen(this.videoElement.nativeElement, 'play', (event) => {
      this.videoHeight = this.videoElement.nativeElement.videoHeight;
      this.videoWidth = this.videoElement.nativeElement.videoWidth;
    });
  }

  handleError(error) {
    console.log('Error: ', error);
    alert('Error: ' + error);
  }
  
  async onVideoCanPlay() {
    this.videoBoundingRect = this.videoElement.nativeElement.getBoundingClientRect();
  }

  // To-do: 카메라 영상 디텍션
  // To-do: 다른 기기 연결하여 디텍션 

  changeModel(event: any) {
    this.model = event.target.value;
  }

  // To-do: 동영상 - 이미지에 따른 디텍션
  // - getDetections의 이미지 - 영상에 따른 분기
  onFileChange(event) {
    if (event.target.files && event.target.files.length > 0) {
      this.svgEnabled = false;
      const file = event.target.files[0];
      const reader = new FileReader();
      const img = new Image();

      img.onload = () => {
        this.showVideo = false;
        this.renderer.setProperty(this.canvas.nativeElement, 'width', this.videoWidth);
        this.renderer.setProperty(this.canvas.nativeElement, 'height', this.videoHeight);
        this.canvas.nativeElement.getContext('2d').drawImage(img, 0, 0)
      };
      img.src = URL.createObjectURL(file);
      reader.readAsDataURL(file);
      reader.onload = () => {
        this.getDetections(reader.result);
      };
    }
  }

  async capture() {
    this.showVideo = false;
    this.renderer.setProperty(this.canvas.nativeElement, 'width', this.videoWidth);
    this.renderer.setProperty(this.canvas.nativeElement, 'height', this.videoHeight);
    this.canvas.nativeElement.getContext('2d').drawImage(this.videoElement.nativeElement, 0, 0)
    
    console.log(this.canvas.nativeElement.toDataURL())
    this.getDetections(this.canvas.nativeElement.toDataURL());
  }

  getDetections(image: any) {
    this.doSpinner = true;
    const formData = new FormData();
    formData.append('image', image);
    formData.append('model', this.model);

    this.http.post(`${this.apiUrl}/dnn/yolo`, formData)
      .subscribe(res => {
        this.currentDetections = res;
        this.svgEnabled = true;
        this.doSpinner = false;
      });
  }
}