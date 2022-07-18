# Failure Cases

Like all methods, XMem can fail. Here, we try to show some illustrative and frankly consistent failure modes that we noticed. We slowed down all videos for visualization.

## Fast motion, similar objects

The first one is fast motion with similarly-looking objects that do not provide sufficient appearance clues for XMem to track. Below is an example from the YouTubeVOS validation set (0e8a6b63bb):

And the source video:

Technically it can be solved by using more positional and motion clues. XMem is not sufficiently proficient at those.

## Shot changes; saliency shift

Ever wondered why I did not include the final scene of Chika Dance when the roach flies off? Because it failed there.

XMem seems to be attracted to any new salient object in the scene when the (true) target object is missing. By new I mean an object that did not appear (or had a different appearance) earlier in the video -- as XMem could not have a memory representation for that object. This happens a lot if the camera shot changes.

Note that the first shot change is not as problematic.
