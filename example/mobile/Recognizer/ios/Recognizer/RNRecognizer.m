#import "RNRecognizer.h"
#import "Recognizer/Recognizer.h"

extern RecognizerModel *reco;

@implementation RNRecognizer

- (dispatch_queue_t) methodQueue
{
  return dispatch_get_main_queue();
}

+ (BOOL)requiresMainQueueSetup
{
  return NO;
}

RCT_EXPORT_MODULE()

RCT_EXPORT_METHOD(Recognize:(NSString *)path resolve:(RCTPromiseResolveBlock)resolve reject: (RCTPromiseRejectBlock)reject) {
dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSError *error;
    @autoreleasepool {
      NSString *res = RecognizerRunFile(reco, path, 10, &error);
      if (!res || error != NULL) {
        NSLog(@"Failed to recognize: %@\n", [error localizedDescription]);
        reject(@"recognize", @"Failed to recognize", NULL);
      }else {
        resolve(res);
      }
    }
  });
}

@end
